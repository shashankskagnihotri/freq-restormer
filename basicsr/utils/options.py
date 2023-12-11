import yaml
from collections import OrderedDict
from os import path as osp


def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


def parse(opt_path, is_train=True, args=None):
    """Parse option file.

    Args:
        opt_path (str): Option file path.
        is_train (str): Indicate whether in training or not. Default: True.

    Returns:
        (dict): Options.
    """
    with open(opt_path, mode='r') as f:
        Loader, _ = ordered_yaml()
        opt = yaml.load(f, Loader=Loader)

    opt['is_train'] = is_train if not args.is_val else False
    opt['train']['adversarial']=args.adversarial
    opt['network_g']['flc_pooling']=args.flc
    opt['network_g']['use_conv'] = args.use_conv
    opt['network_g']['use_alpha'] = args.use_alpha
    #if args.learn_alpha:
    opt['network_g']['learn_alpha'] = args.learn_alpha
    opt['network_g']['drop_alpha'] = args.drop_alpha
    opt['network_g']['first_drop_alpha'] = args.first_drop_alpha
    opt['network_g']['test_wo_drop_alpha'] = args.test_wo_drop_alpha
    opt['network_g']['use_blur'] = args.blur
    opt['network_g']['kernel_size']=args.kernel_size
    opt['network_g']['para_kernel_size']=args.para_kernel_size
    opt['network_g']['padding'] = "constant" if args.zero_padding else "reflect"
    opt['network_g']['upsampling_method'] = args.upsample_method
    opt['attack']['adv_attack'] = args.adv_attack

    if args.adv_attack:
        opt['attack']['method'] = args.attack_method
        opt['attack']['iterations'] = args.attack_iterations
        opt['attack']['alpha'] = args.attack_alpha
    if args.resume_state != None:
        opt['path']['resume_state'] = args.resume_state
    if args.pretrain_network_g != None:
        opt['path']['pretrain_network_g'] = args.pretrain_network_g

    # datasets
    for phase, dataset in opt['datasets'].items():
        # for several datasets, e.g., test_1, test_2
        phase = phase.split('_')[0]
        dataset['phase'] = phase
        if 'scale' in opt:
            dataset['scale'] = opt['scale']
        if dataset.get('dataroot_gt') is not None:
            dataset['dataroot_gt'] = osp.expanduser(dataset['dataroot_gt'])
        if dataset.get('dataroot_lq') is not None:
            dataset['dataroot_lq'] = osp.expanduser(dataset['dataroot_lq'])

    # paths
    for key, val in opt['path'].items():
        if (val is not None) and ('resume_state' in key
                                  or 'pretrain_network' in key):
            opt['path'][key] = osp.expanduser(val)
    opt['path']['root'] = osp.abspath(
        osp.join(__file__, osp.pardir, osp.pardir, osp.pardir))
    #'with' if args.drop_alpha else 'without'
    if args.use_conv:
        concat_word = "conv"
    else:
        concat_word = "concat"

    if args.first_drop_alpha:
        drop_word = "first_layer"
    elif args.drop_alpha:
        drop_word = "with"
    else:
        drop_word = "without"
    if is_train:
        if args.kernel_size >2:            
            opt['name'] += '_flc_pooling_{}_Adversarial_training_{}_trans_kernel_{}_para_trans_kernel_{}_upsampling_{}{}'.format(opt['network_g']['flc_pooling'], opt['train']['adversarial'], opt['network_g']['kernel_size'], opt['network_g']['para_kernel_size'], args.upsample_method, '_normal_training' if args.zero_padding else '')
        else:
            opt['name'] += '_flc_pooling_low_freq_{}_{}_alpha_{}_{}_blur_{}_drop_alpha_Adversarial_training_{}_pixel_shuffle_{}_padding_upsampling_{}'.format(opt['network_g']['flc_pooling'], concat_word, 'learned' if args.learn_alpha else 'random', 'with' if args.blur else 'without', drop_word, opt['train']['adversarial'], "zero" if args.zero_padding else "mirror", args.upsample_method)
        experiments_root = osp.join(opt['path']['root'], 'experiments',
                                    opt['name'])
        if args.debugging:
            experiments_root = osp.join(opt['path']['root'], 'debugging',
                                    opt['name'])
        opt['path']['experiments_root'] = experiments_root
        opt['path']['models'] = osp.join(experiments_root, 'models')
        opt['path']['training_states'] = osp.join(experiments_root,
                                                  'training_states')
        opt['path']['log'] = experiments_root
        opt['path']['visualization'] = osp.join(experiments_root,
                                                'visualization')

        # change some options for debug mode
        if 'debug' in opt['name']:
            if 'val' in opt:
                opt['val']['val_freq'] = 8
            opt['logger']['print_freq'] = 1
            opt['logger']['save_checkpoint_freq'] = 8
    else:  # test
        if args.kernel_size > 2:            
            results_root = osp.join(opt['path']['root'], '_flc_pooling_{}_trans_kernel_{}_para_trans_kernel_{}_upsampling_{}{}'.format(opt['train']['flc'], opt['network_g']['kernel_size'], opt['network_g']['para_kernel_size'], args.upsample_method, '_larger' if args.zero_padding else ''), 'Adversarial_training_{}'.format(opt['train']['adversarial']), opt['attack']['method'], opt['attack']['iterations'], opt['attack']['alpha'], opt['attack']['epsilon'],  'results', opt['name'])
        elif args.use_alpha:
            results_root = osp.join(opt['path']['root'], 'new', '_flc_pooling_low_freq_{}_{}_alpha_{}_blurring_{}_padding_upsampling_{}'.format(opt['train']['flc'], concat_word, 'learned' if args.learn_alpha else 'random', 'with' if args.blur else 'without', args.upsample_method), 'Adversarial_training_{}'.format(opt['train']['adversarial']), opt['attack']['method'], opt['attack']['iterations'], opt['attack']['alpha'], opt['attack']['epsilon'], 'results', opt['name'], "zero" if args.zero_padding else "mirror")
        else:
            results_root = osp.join(opt['path']['root'], 'new',  '_flc_pooling_low_freq_{}_upsampling_{}'.format(opt['train']['flc'],  args.upsample_method), 'Adversarial_training_{}'.format(opt['train']['adversarial']), opt['attack']['method'], opt['attack']['iterations'], opt['attack']['alpha'], opt['attack']['epsilon'],  'results', opt['name'])
        opt['path']['results_root'] = results_root
        opt['path']['log'] = results_root
        opt['path']['visualization'] = osp.join(results_root, 'visualization')

        print("Results stored in: {}".format(opt['path']['results_root']))
    return opt


def dict2str(opt, indent_level=1):
    """dict to string for printing options.

    Args:
        opt (dict): Option dict.
        indent_level (int): Indent level. Default: 1.

    Return:
        (str): Option string for printing.
    """
    msg = '\n'
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_level * 2) + k + ':['
            msg += dict2str(v, indent_level + 1)
            msg += ' ' * (indent_level * 2) + ']\n'
        else:
            msg += ' ' * (indent_level * 2) + k + ': ' + str(v) + '\n'
    return msg
