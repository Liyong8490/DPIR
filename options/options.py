import os
import json
from collections import OrderedDict
from datetime import datetime


class NoneDict(dict):
    def __missing__(self, key):
        return None


def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')


def parse(opt_path, is_train=True): # remove comments starting with '//'
    json_str = ''
    with open(opt_path, 'r') as f:
        for line in f:
            line = line.split('//')[0] + '\n'
            json_str += line

    opt = json.loads(json_str, object_pairs_hook=OrderedDict)

    opt['timestamp'] = get_timestamp()
    opt['is_train'] = is_train

    # datasets
    for phase, dataset in opt['datasets_opt'].items():
        phase = phase.split('_')[0]
        dataset['phase'] = phase
        is_lmdb = False
        if 'label_root' in dataset and dataset['label_root'] is not None:
            dataset['label_root'] = os.path.expanduser(dataset['label_root'])
            if dataset['label_root'].endswith('lmdb'):
                is_lmdb = True
        if 'noisy_root' in dataset and dataset['noisy_root'] is not None:
            dataset['noisy_root'] = os.path.expanduser(dataset['noisy_root'])
        dataset['data_type'] = 'lmdb' if is_lmdb else 'img'

    for key, path in opt['path_opt'].items():
        if path and key in opt['path_opt']:
            opt['path_opt'][key] = os.path.expanduser(path)
    if is_train:
        experiments_root = os.path.join(opt['path_opt']['root'], 'experiments', opt['name'])
        opt['path_opt']['experiments_root'] = experiments_root
        opt['path_opt']['log'] = experiments_root
        opt['path_opt']['models'] = os.path.join(experiments_root, 'models')
        opt['path_opt']['val_images'] = os.path.join(experiments_root, 'val_images')
        # change some options for debug mode
        if 'debug' in opt['name']:
            opt['train_opt']['val_freq'] = 8
            opt['logger_opt']['print_freq'] = 2
            opt['logger_opt']['save_cpkt_freq'] = 8
            opt['train_opt']['lr_decay_iter'] = 10
    else:  # test
        results_root = os.path.join(opt['path_opt']['root'], 'results', opt['name'])
        opt['path_opt']['results_root'] = results_root
        opt['path_opt']['log'] = results_root

    # export CUDA_VISIBLE_DEVICES
    gpu_list = ','.join(str(x) for x in opt['gpu_ids'])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    print("export CUDA_VISIBLE_DEVICES=" + gpu_list)
    return opt


def save(opt):
    dump_dir = opt['path_opt']['experiments_root'] if opt['is_train'] else opt['path_opt']['results_root']
    dump_path = os.path.join(dump_dir, 'options.json')
    with open(dump_path, 'w') as f:
        json.dump(opt, f, indent=2)


def dict_to_nonedict(opt):
    """
    convert to NoneDict, which return None for missing key.
    """
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt
