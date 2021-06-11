import argparse
from os import path as osp

from basicsr.utils import scandir
from basicsr.utils.lmdb_util import make_lmdb_from_imgs, make_lmdb_from_flows


def create_lmdb_for_div2k():
    """Create lmdb files for DIV2K dataset.

    Usage:
        Before run this script, please run `extract_subimages.py`.
        Typically, there are four folders to be processed for DIV2K dataset.
            DIV2K_train_HR_sub
            DIV2K_train_LR_bicubic/X2_sub
            DIV2K_train_LR_bicubic/X3_sub
            DIV2K_train_LR_bicubic/X4_sub
        Remember to modify opt configurations according to your settings.
    """
    # HR images
    folder_path = 'datasets/DIV2K/DIV2K_train_HR_sub'
    lmdb_path = 'datasets/DIV2K/DIV2K_train_HR_sub.lmdb'
    img_path_list, keys = prepare_keys_div2k(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    # LRx2 images
    folder_path = 'datasets/DIV2K/DIV2K_train_LR_bicubic/X2_sub'
    lmdb_path = 'datasets/DIV2K/DIV2K_train_LR_bicubic_X2_sub.lmdb'
    img_path_list, keys = prepare_keys_div2k(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    # LRx3 images
    folder_path = 'datasets/DIV2K/DIV2K_train_LR_bicubic/X3_sub'
    lmdb_path = 'datasets/DIV2K/DIV2K_train_LR_bicubic_X3_sub.lmdb'
    img_path_list, keys = prepare_keys_div2k(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    # LRx4 images
    folder_path = 'datasets/DIV2K/DIV2K_train_LR_bicubic/X4_sub'
    lmdb_path = 'datasets/DIV2K/DIV2K_train_LR_bicubic_X4_sub.lmdb'
    img_path_list, keys = prepare_keys_div2k(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)


def prepare_keys_div2k(folder_path):
    """Prepare image path list and keys for DIV2K dataset.

    Args:
        folder_path (str): Folder path.

    Returns:
        list[str]: Image path list.
        list[str]: Key list.
    """
    print('Reading image path list ...')
    img_path_list = sorted(
        list(scandir(folder_path, suffix='png', recursive=False)))
    keys = [img_path.split('.png')[0] for img_path in sorted(img_path_list)]

    return img_path_list, keys


def create_lmdb_for_reds():
    """Create lmdb files for REDS dataset.

    Usage:
        Before run this script, please run `merge_reds_train_val.py`.
        We take two folders for example:
            train_sharp
            train_sharp_bicubic
        Remember to modify opt configurations according to your settings.
    """
    # # train_sharp
    # folder_path = '/cluster/work/cvl/yawli/data/REDS/train_sharp'
    # lmdb_path = '/cluster/work/cvl/yawli/data/REDS/train_sharp_with_val.lmdb'
    # img_path_list, keys = prepare_keys_reds(folder_path)
    # make_lmdb_from_imgs(
    #     folder_path, lmdb_path, img_path_list, keys, multiprocessing_read=True)
    #
    # # train_sharp_bicubic
    # folder_path = '/cluster/work/cvl/yawli/data/REDS/train_sharp_bicubic/X4'
    # lmdb_path = '/cluster/work/cvl/yawli/data/REDS/train_sharp_bicubic_with_val.lmdb'
    # img_path_list, keys = prepare_keys_reds(folder_path)
    # make_lmdb_from_imgs(
    #     folder_path, lmdb_path, img_path_list, keys, multiprocessing_read=True)

    # train_sharp_bicubic_flow
    folder_path = '/cluster/work/cvl/videosr/REDS/train_sharp_bicubic_flow/X4'
    lmdb_path = '/cluster/work/cvl/videosr/REDS/train_sharp_bicubic_flow_with_val.lmdb'
    img_path_list, keys = prepare_keys_reds(folder_path, 'flo')
    make_lmdb_from_flows(
        folder_path, lmdb_path, img_path_list, keys, multiprocessing_read=True)


def prepare_keys_reds(folder_path, suffix='png'):
    """Prepare image path list and keys for REDS dataset.

    Args:
        folder_path (str): Folder path.

    Returns:
        list[str]: Image path list.
        list[str]: Key list.
    """
    print('Reading image path list ...')
    img_path_list = sorted(
        list(scandir(folder_path, suffix=suffix, recursive=True)))
    keys = [v.split(f'.{suffix}')[0] for v in img_path_list]  # example: 000/00000000

    return img_path_list, keys


def create_lmdb_for_vimeo90k():
    """Create lmdb files for Vimeo90K dataset.

    Usage:
        Remember to modify opt configurations according to your settings.
    """
    # # GT (only4th)
    # folder_path = '/cluster/work/cvl/videosr/vimeo90k/vimeo_septuplet/sequences'
    # lmdb_path = '/cluster/work/cvl/videosr/vimeo90k/vimeo90k_train_GT_only4th.lmdb'
    # train_list_path = '/cluster/work/cvl/videosr/vimeo90k/vimeo_septuplet/sep_trainlist.txt'
    # img_path_list, keys = prepare_keys_vimeo90k(folder_path, train_list_path, 'gt')
    # make_lmdb_from_imgs(
    #     folder_path, lmdb_path, img_path_list, keys, multiprocessing_read=True)
    
    # # GT (all)
    folder_path = '/cluster/work/cvl/videosr/vimeo90k/vimeo_septuplet/sequences'
    lmdb_path = '/cluster/work/cvl/videosr/vimeo90k/vimeo90k_train_GT_all.lmdb'
    train_list_path = '/cluster/work/cvl/videosr/vimeo90k/vimeo_septuplet/sep_trainlist.txt'
    img_path_list, keys = prepare_keys_vimeo90k(folder_path, train_list_path, 'gt-all')
    make_lmdb_from_imgs(
        folder_path, lmdb_path, img_path_list, keys, multiprocessing_read=True)

    # # LQ
    # folder_path = '/cluster/work/cvl/videosr/vimeo90k/vimeo_septuplet_matlabLRx4/sequences'
    # lmdb_path = '/cluster/work/cvl/videosr/vimeo90k/vimeo90k_train_LR7frames.lmdb'
    # train_list_path = '/cluster/work/cvl/videosr/vimeo90k/vimeo_septuplet/sep_trainlist.txt'
    # img_path_list, keys = prepare_keys_vimeo90k(folder_path, train_list_path,
    #                                             'lq')
    # make_lmdb_from_imgs(
    #     folder_path, lmdb_path, img_path_list, keys, multiprocessing_read=True)

    # flow
    # folder_path = '/cluster/work/cvl/videosr/vimeo90k/vimeo_septuplet_matlabLRx4_flow/sequences'
    # lmdb_path = '/cluster/work/cvl/videosr/vimeo90k/vimeo90k_train_LR_flow.lmdb'
    # train_list_path = '/cluster/work/cvl/videosr/vimeo90k/vimeo_septuplet/sep_trainlist.txt'
    # img_path_list, keys = prepare_keys_vimeo90k(folder_path, train_list_path,
    #                                             'flow')
    # make_lmdb_from_flows(
    #     folder_path, lmdb_path, img_path_list, keys, multiprocessing_read=True)

def prepare_keys_vimeo90k(folder_path, train_list_path, mode):
    """Prepare image path list and keys for Vimeo90K dataset.

    Args:
        folder_path (str): Folder path.
        train_list_path (str): Path to the official train list.
        mode (str): One of 'gt' or 'lq'.

    Returns:
        list[str]: Image path list.
        list[str]: Key list.
    """
    print('Reading image path list ...')
    with open(train_list_path, 'r') as fin:
        train_list = [line.strip() for line in fin]

    img_path_list = []
    keys = []
    if mode == 'flow':
        for line in train_list:
            folder, subfolder = line.split('/')
            suffix = [f'p{i}' for i in range(3, 0, -1)] + [f'n{i}' for i in range(1, 4)]
            img_path_list.extend(
                [osp.join(folder, subfolder, f'im4_{s}.flo') for s in suffix]
            )
            keys.extend([f'{folder}/{subfolder}/im4_{s}' for s in suffix])
    else:
        for line in train_list:
            folder, sub_folder = line.split('/')
            img_path_list.extend(
                [osp.join(folder, sub_folder, f'im{j + 1}.png') for j in range(7)])
            keys.extend([f'{folder}/{sub_folder}/im{j + 1}' for j in range(7)])

        if mode == 'gt':
            print('Only keep the 4th frame for the gt mode.')
            img_path_list = [v for v in img_path_list if v.endswith('im4.png')]
            keys = [v for v in keys if v.endswith('/im4')]

    return img_path_list, keys


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dataset',
        type=str,
        help=(
            "Options: 'DIV2K', 'REDS', 'Vimeo90K' "
            'You may need to modify the corresponding configurations in codes.'
        ))
    args = parser.parse_args()
    dataset = args.dataset.lower()
    if dataset == 'div2k':
        create_lmdb_for_div2k()
    elif dataset == 'reds':
        create_lmdb_for_reds()
    elif dataset == 'vimeo90k':
        create_lmdb_for_vimeo90k()
    else:
        raise ValueError('Wrong dataset.')
