import random
import torch
from pathlib import Path
from torch.utils import data as data
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.flow_util import flowfrombytes
import pdb


class Vimeo90KDataset(data.Dataset):
    """Vimeo90K dataset for training.

    The keys are generated from a meta info txt file.
    basicsr/data/meta_info/meta_info_Vimeo90K_train_GT.txt

    Each line contains:
    1. clip name; 2. frame number; 3. image shape, seperated by a white space.
    Examples:
        00001/0001 7 (256,448,3)
        00001/0002 7 (256,448,3)

    Key examples: "00001/0001"
    GT (gt): Ground-Truth;
    LQ (lq): Low-Quality, e.g., low-resolution/blurry/noisy/compressed frames.

    The neighboring frame list for different num_frame:
    num_frame | frame list
             1 | 4
             3 | 3,4,5
             5 | 2,3,4,5,6
             7 | 1,2,3,4,5,6,7

    Args:
        opt (dict): Config for train dataset. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.

            num_frame (int): Window size for input frames.
            gt_size (int): Cropped patched size for gt patches.
            random_reverse (bool): Random reverse input frames.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
    """

    def __init__(self, opt):
        super(Vimeo90KDataset, self).__init__()
        self.opt = opt
        self.gt_root, self.lq_root = Path(opt['dataroot_gt']), Path(
            opt['dataroot_lq'])
        self.flow_root = Path(
            opt['dataroot_flow']) if opt['dataroot_flow'] is not None else None
        self.num_frame = opt['num_frame']

        with open(opt['meta_info_file'], 'r') as fin:
            self.keys = [line.split(' ')[0] for line in fin]

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.is_lmdb = False
        if self.io_backend_opt['type'] == 'lmdb':
            self.is_lmdb = True
            if self.flow_root is not None:
                self.io_backend_opt['db_paths'] = [
                    self.lq_root, self.gt_root, self.flow_root
                ]
                self.io_backend_opt['client_keys'] = ['lq', 'gt', 'flow']
            else:
                self.io_backend_opt['db_paths'] = [self.lq_root, self.gt_root]
                self.io_backend_opt['client_keys'] = ['lq', 'gt']

        # indices of input images
        self.neighbor_list = [
            i + (9 - opt['num_frame']) // 2 for i in range(opt['num_frame'])     # [1, 2, 3, 4, 5, 6, 7]
        ]

        # temporal augmentation configs
        self.random_reverse = opt['random_reverse']
        logger = get_root_logger()
        logger.info(f'Random reverse is {self.random_reverse}.')

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # random reverse
        if self.random_reverse and random.random() < 0.5:
            self.neighbor_list.reverse()

        scale = self.opt['scale']
        gt_size = self.opt['gt_size']
        key = self.keys[index]
        clip, seq = key.split('/')  # key example: 00001/0001

        # get the GT frame (im4.png)
        if self.opt['all_gt']:
            img_gts = []
            for neighbor in self.neighbor_list:
                if self.is_lmdb:
                    img_gt_path = f'{key}/im{neighbor}'
                else:
                    img_gt_path = self.gt_root / clip / seq / f'im{neighbor}.png'
                img_bytes = self.file_client.get(img_gt_path, 'gt')
                img_gt = imfrombytes(img_bytes, float32=True)
                img_gts.append(img_gt)
        else:
            if self.is_lmdb:
                img_gt_path = f'{key}/im4'
            else:
                img_gt_path = self.gt_root / clip / seq / 'im4.png'
            img_bytes = self.file_client.get(img_gt_path, 'gt')
            img_gt = imfrombytes(img_bytes, float32=True)

        # get the neighboring LQ frames
        img_lqs = []
        for neighbor in self.neighbor_list:
            if self.is_lmdb:
                img_lq_path = f'{clip}/{seq}/im{neighbor}'
            else:
                img_lq_path = self.lq_root / clip / seq / f'im{neighbor}.png'
            img_bytes = self.file_client.get(img_lq_path, 'lq')
            img_lq = imfrombytes(img_bytes, float32=True)
            img_lqs.append(img_lq)

        # get flows
        if self.flow_root is not None:
            img_flows = []
            # read previous flows
            for i in range(self.opt['num_frame'] // 2, 0, -1):
                if self.is_lmdb:
                    flow_path = f'{clip}/{seq}/im4_p{i}'
                else:
                    flow_path = (
                        self.flow_root / clip / seq / f'im4_p{i}.flo')
                img_bytes = self.file_client.get(flow_path, 'flow')
                flow = flowfrombytes(img_bytes, self.is_lmdb, img_lqs[0].shape)
                img_flows.append(flow)
            # read next flows
            for i in range(1, self.opt['num_frame'] // 2 + 1):
                if self.is_lmdb:
                    flow_path = f'{clip}/{seq}/im4_n{i}'
                else:
                    flow_path = (
                        self.flow_root / clip / seq / f'im4_n{i}.flo')
                img_bytes = self.file_client.get(flow_path, 'flow')
                flow = flowfrombytes(img_bytes, self.is_lmdb, img_lqs[0].shape)
                img_flows.append(flow)

            # for random crop, here, img_flows and img_lqs have the same
            # spatial size
            img_lqs.extend(img_flows)

        # randomly crop
        if self.opt['all_gt']:
            img_gts, img_lqs = paired_random_crop(img_gts, img_lqs, gt_size, scale, img_gt_path)
        else:
            img_gt, img_lqs = paired_random_crop(img_gt, img_lqs, gt_size, scale, img_gt_path)
        if self.flow_root is not None:
            img_lqs, img_flows = img_lqs[:self.opt['num_frame']], img_lqs[self.opt['num_frame']:]

        # augmentation - flip, rotate
        if self.opt['all_gt']:
            img_lqs.extend(img_gts)
        else:
            img_lqs.append(img_gt)
        if self.flow_root is not None:
            img_results, img_flows = augment(img_lqs, self.opt['use_flip'], self.opt['use_rot'], img_flows)
        else:
            img_results = augment(img_lqs, self.opt['use_flip'], self.opt['use_rot'])

        img_results = img2tensor(img_results)

        if self.opt['all_gt']:
            img_lqs = torch.stack(img_results[:self.num_frame], dim=0)
            img_gts = torch.stack(img_results[self.num_frame:], dim=0)
        else:
            img_lqs = torch.stack(img_results[0:-1], dim=0)
            img_gt = img_results[-1]

        if self.flow_root is not None:
            img_flows = img2tensor(img_flows)
            # add the zero center flow
            img_flows.insert(self.opt['num_frame'] // 2,
                             torch.zeros_like(img_flows[0]))
            img_flows = torch.stack(img_flows, dim=0)

        # img_lqs: (t, c, h, w)
        # img_gt: (c, h, w)
        # img_gts: (t, c, h, w)
        # key: str

        if self.flow_root is not None:
            return {'lq': img_lqs, 'flow': img_flows, 'gt': img_gt, 'key': key}
        elif self.opt['all_gt']:
            return {'lq': img_lqs, 'gt': img_gts, 'key': key}
        else:
            return {'lq': img_lqs, 'gt': img_gt, 'key': key}

    def __len__(self):
        return len(self.keys)
