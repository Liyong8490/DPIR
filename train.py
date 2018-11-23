import os
import sys
import math
import argparse
import time
import random
import torch
import options.options as option
from utils import util
from data import create_dataloader, create_dataset
from collections import OrderedDict
from models import create_model
from utils.logger import Logger, PrintLogger


def main():
    parser = argparse.ArgumentParser("DPIR Deep Prior based Image Restoration.")
    parser.add_argument('-opt', type=str, default='options/train/denoise_dpir.json', help="Path to option JSON file.")
    opt = option.parse(parser.parse_args().opt, is_train=True)

    util.mkdir_and_rename(opt['path_opt']['experiments_root'])
    util.mkdirs((path for key, path in opt['path_opt'].items()
                 if not key == 'experiments_root' and 'pretrained_model' not in key))
    option.save(opt)
    opt = option.dict_to_nonedict(opt)

    sys.stdout = PrintLogger(opt['path_opt']['log'])

    seed = opt['train_opt']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    print("Random Seed: ", seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # create train and val dataloader
    for phase, dataset_opt in opt['datasets_opt'].items():
        if phase == 'train':
            train_set = create_dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))
            print('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
            total_iters = int(opt['train_opt']['max_iter'])
            total_epoches = int(math.ceil(total_iters / train_size))
            print('Total epoches needed: {:d} for iters {:,d}'.format(total_epoches, total_iters))
            train_loader = create_dataloader(train_set, dataset_opt)
        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt)
            print('Number of val images in [{:s}]: {:d}'.format(dataset_opt['name'], len(val_set)))
        else:
            raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))
    assert train_loader is not None

    # Create model
    model = create_model(opt)
    # create logger
    logger = Logger(opt)

    current_step = 0
    start_time = time.time()
    print('---------- Start training -------------')
    for epoch in range(total_epoches):
        for i, train_data in enumerate(train_loader):
            current_step += 1
            if current_step > total_iters:
                break

            # training
            model.feed_data(train_data)
            model.optimize_parameters()

            time_elapsed = time.time() - start_time
            start_time = time.time()

            # log
            if current_step % opt['logger_opt']['print_freq'] == 0:
                logs = model.get_current_log()
                print_rlt = OrderedDict()
                print_rlt['model'] = opt['model']
                print_rlt['epoch'] = epoch
                print_rlt['iters'] = current_step
                print_rlt['time'] = time_elapsed
                for k, v in logs.items():
                    print_rlt[k] = v
                print_rlt['lr'] = model.get_current_learning_rate()
                logger.print_format_results('train', print_rlt)

            # save models
            if current_step % opt['logger_opt']['save_ckpt_freq'] == 0:
                print('Saving the model at the end of iter {:d}.'.format(current_step))
                model.save(current_step)

            # validation
            if current_step % opt['train_opt']['val_freq'] == 0:
                print('---------- validation -------------')
                start_time = time.time()

                avg_psnr = 0.0
                idx = 0
                for val_data in val_loader:
                    idx += 1
                    img_name = os.path.splitext(os.path.basename(val_data['NI_path'][0]))[0]
                    img_dir = os.path.join(opt['path_opt']['val_images'], img_name)
                    util.mkdir(img_dir)

                    model.feed_data(val_data)
                    model.test_embed()

                    visuals = model.get_current_visuals()
                    sr_img = util.tensor2img(visuals['preds'], min_max=(0, 1))  # uint8
                    gt_img = util.tensor2img(visuals['labels'], min_max=(0, 1))  # uint8

                    # Save SR images for reference
                    save_img_path = os.path.join(img_dir, '{:s}_{:d}.png'.format( \
                        img_name, current_step))
                    util.save_img(sr_img, save_img_path)

                    # calculate PSNR
                    # crop_size = opt['scale']
                    # cropped_sr_img = sr_img[crop_size:-crop_size, crop_size:-crop_size, :]
                    # cropped_gt_img = gt_img[crop_size:-crop_size, crop_size:-crop_size, :]
                    # avg_psnr += util.psnr(cropped_sr_img, cropped_gt_img)
                    avg_psnr += util.psnr(sr_img, gt_img)

                avg_psnr = avg_psnr / idx
                time_elapsed = time.time() - start_time
                # Save to log
                print_rlt = OrderedDict()
                print_rlt['model'] = opt['model']
                print_rlt['epoch'] = epoch
                print_rlt['iters'] = current_step
                print_rlt['time'] = time_elapsed
                print_rlt['psnr'] = avg_psnr
                logger.print_format_results('val', print_rlt)
                print('-----------------------------------')

            # update learning rate
            model.update_learning_rate()

    print('Saving the final model.')
    model.save('latest')
    print('End of training.')


if __name__ == '__main__':
    # # OpenCV get stuck in transform when used in DataLoader
    # # https://github.com/pytorch/pytorch/issues/1838
    # # However, cause problem reading lmdb
    # import torch.multiprocessing as mp
    # mp.set_start_method('spawn', force=True)
    main()

