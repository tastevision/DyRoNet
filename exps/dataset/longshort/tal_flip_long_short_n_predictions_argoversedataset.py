import cv2
import numpy as np
# import json
# import time
# from PIL import Image
from pycocotools.coco import COCO
# from collections import defaultdict

# import io
import os

# from yolox.data.dataloading import get_yolox_datadir
from yolox.data.datasets.datasets_wrapper import Dataset

# from loguru import logger

class LONGSHORT_xN_ARGOVERSEDataset(Dataset):
    """
    COCO dataset class.
    """
    def __init__(self, data_dir='/data/Datasets/', json_file='train.json',
                 name='train', img_size=(416,416), preproc=None, cache=False,
                 short_cfg=dict(
                    frame_num=2,
                    delta=1),
                 long_cfg=dict(
                    frame_num=2,
                    delta=2,
                    include_current_frame=True),
                 prediction_step=1,
                 origin_support_step=False,
                 ):
        """
        COCO dataset initialization. Annotation data are read into memory by COCO API.
        Args:
            data_dir (str): dataset root directory
            json_file (str): COCO json file name
            name (str): COCO data name (e.g. 'train2017' or 'val2017')
            img_size (int): target image size after pre-processing
            preproc: data augmentation strategy
            debug (bool): if True, only one data id is selected from the dataset
        """
        super().__init__(img_size)
        self.data_dir = data_dir
        self.json_file = json_file
        self.coco = COCO(self.data_dir+'/Argoverse-HD/annotations/'+self.json_file)
        self.ids = self.coco.getImgIds()
        self.seq_dirs = self.coco.dataset['seq_dirs']
        self.class_ids = sorted(self.coco.getCatIds())
        # {0: {'id': 0, 'name': 'person'}, 1: {'id': 1, 'name': 'bicycle'}, 2: {'id': 2, 'name': 'car'},
        # 3: {'id': 3, 'name': 'motorcycle'}, 4: {'id': 4, 'name': 'bus'}, 5: {'id': 5, 'name': 'truck'},
        # 6: {'id': 6, 'name': 'traffic_light'}, 7: {'id': 7, 'name': 'stop_sign'}}
        self._classes = self.coco.cats
        self.name = name
        self.max_labels = 50
        self.img_size = img_size
        self.preproc = preproc
        self.short_cfg = short_cfg
        self.long_cfg = long_cfg
        self.prediction_step = prediction_step
        self.origin_support_step = origin_support_step
        self.annotations = self._load_coco_annotations()
        self.imgs = None

    def __len__(self):
        return len(self.ids)

    def __del__(self):
        if self.imgs:
            del self.imgs

    def _get_im_anno_file_name_same(self, id_):
        short_im_annos = dict()
        short_file_names = dict()
        long_im_annos = dict()
        long_file_names = dict()
        im_ann = self.coco.loadImgs(id_)[0]
        im_name = im_ann['name']
        im_sid = im_ann['sid']

        # short
        for i in range(self.short_cfg["frame_num"]):
            short_im_annos[f'im_anno_{i*self.short_cfg["delta"]}'] = im_ann
            short_file_names[f'im_anno_{i*self.short_cfg["delta"]}'] = \
                os.path.join(self.data_dir, 'Argoverse-1.1', 'tracking', self.seq_dirs[im_sid], im_name)
        # long
        for i in range(self.long_cfg["frame_num"]):
            cur_id = i if self.long_cfg["include_current_frame"] else i+1
            long_im_annos[f'im_anno_{cur_id*self.long_cfg["delta"]}'] = im_ann
            long_file_names[f'im_anno_{cur_id*self.long_cfg["delta"]}'] = \
                os.path.join(self.data_dir, 'Argoverse-1.1', 'tracking', self.seq_dirs[im_sid], im_name)

        return short_im_annos, long_im_annos, short_file_names, long_file_names

    def _get_im_anno_file_name_diff(self, id_):
        short_im_annos = dict()
        short_file_names = dict()
        long_im_annos = dict()
        long_file_names = dict()
        # short 和 long 支路最大的delta
        max_delta = (max((self.short_cfg["frame_num"]-1)*self.short_cfg["delta"], (self.long_cfg["frame_num"]-1)*self.long_cfg["delta"]) 
                    if self.long_cfg["include_current_frame"] else 
                    max((self.short_cfg["frame_num"]-1)*self.short_cfg["delta"], (self.long_cfg["frame_num"])*self.long_cfg["delta"]))

        # 获取边界的delta
        bound_delta = self._get_boundary_delta(id_, max_delta)

        # short
        for i in range(self.short_cfg["frame_num"]):
            cur_delta = min(i*self.short_cfg["delta"], bound_delta)
            cur_im_ann = self.coco.loadImgs(id_ - cur_delta)[0]
            cur_im_name = cur_im_ann['name']
            cur_im_sid = cur_im_ann['sid']
            short_im_annos[f'im_anno_{i*self.short_cfg["delta"]}'] = cur_im_ann
            short_file_names[f'im_anno_{i*self.short_cfg["delta"]}'] = \
                os.path.join(self.data_dir, 'Argoverse-1.1', 'tracking', self.seq_dirs[cur_im_sid], cur_im_name)
        # long
        for i in range(self.long_cfg["frame_num"]):
            cur_id = i if self.long_cfg["include_current_frame"] else i+1
            cur_delta = min(cur_id*self.long_cfg["delta"], bound_delta)
            cur_im_ann = self.coco.loadImgs(id_ - cur_delta)[0]
            cur_im_name = cur_im_ann['name']
            cur_im_sid = cur_im_ann['sid']
            long_im_annos[f'im_anno_{cur_id*self.long_cfg["delta"]}'] = cur_im_ann
            long_file_names[f'im_anno_{cur_id*self.long_cfg["delta"]}'] = \
                os.path.join(self.data_dir, 'Argoverse-1.1', 'tracking', self.seq_dirs[cur_im_sid], cur_im_name)

        return short_im_annos, long_im_annos, short_file_names, long_file_names

    def _get_boundary_delta(self, id_, max_delta):
        """ 获取边界delta，后面超出此边界的帧需要特殊处理
        """
        if id_ - max_delta < 0: # 第一个视频，max_delta超出边界
            bound_delta = id_
        elif self.coco.loadImgs(id_)[0]["sid"] != self.coco.loadImgs(id_-max_delta)[0]["sid"]: # 其它视频，max_delta超出边界
            for i in range(max_delta-1, 0, -1):
                if self.coco.loadImgs(id_)[0]["sid"] == self.coco.loadImgs(id_-i)[0]["sid"]:
                    bound_delta = i
                    break
        else:
            bound_delta = float('inf')

        return bound_delta

    def _get_target_im_id(self, id_):
        seq_len = len(self.ids)
        p_step = self.prediction_step
        # target im 为最后一段视频最后一帧
        if id_ in set([seq_len-i for i in range(1, p_step+2)]):
            target_img_id = int(seq_len) - 1
        else:
            # target im 为每一段视频的起始帧
            if self.coco.dataset['images'][int(id_)]['fid'] == 0:
                target_img_id = int(id_)
            # target im 位于下一段视频
            elif (self.coco.dataset['images'][int(id_)]['fid'] > 
                    self.coco.dataset['images'][int(id_)+p_step]['fid']):
                target_img_id = int(id_) + p_step - (self.coco.dataset['images'][int(id_)+p_step]['fid'] + 1)
            # 一般情况，target im 为当前id_之后的p_step帧
            else:
                target_img_id = int(id_) + p_step
        return target_img_id

    def _load_coco_annotations(self):
        return [self.load_anno_from_ids(_ids) for _ids in self.ids]

    def load_anno(self, index):
        return self.annotations[index][0]

    def load_anno_from_ids(self, id_):
        im_ann = self.coco.loadImgs(id_)[0]
        width = im_ann["width"]
        height = im_ann["height"]
        im_name = im_ann['name']
        im_sid = im_ann['sid']

        seq_len = len(self.ids)

        # images
        ## front fid
        if self.coco.dataset['images'][int(id_)]['fid'] == 0:
            short_im_annos, long_im_annos, short_file_names, long_file_names = \
                self._get_im_anno_file_name_same(id_)

        ## back seq fid
        elif int(id_) == seq_len-1:
            short_im_annos, long_im_annos, short_file_names, long_file_names = \
                self._get_im_anno_file_name_same(id_)
        
        ## back fid
        elif self.coco.dataset['images'][int(id_ + 1)]['fid'] == 0:
            short_im_annos, long_im_annos, short_file_names, long_file_names = \
                self._get_im_anno_file_name_same(id_)

        else:
            short_im_annos, long_im_annos, short_file_names, long_file_names = \
                self._get_im_anno_file_name_diff(id_)
            # im_ann_support = self.coco.loadImgs(id_ - 1)[0]

        target_img_id = self._get_target_im_id(id_)
        anno_ids = self.coco.getAnnIds(imgIds=[target_img_id], iscrowd=False)

        # # annotations
        # ## back seq fid
        # if id_ in [seq_len-1, seq_len-2]:
        #     anno_ids = self.coco.getAnnIds(imgIds=[int(seq_len)], iscrowd=False) # bug, should be imgIds=[int(seq_len-1)]
        # ## back fid
        # else:
        #     if self.coco.dataset['images'][int(id_)]['fid'] == 0:
        #         anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)

        # ## front fid
        #     elif self.coco.dataset['images'][int(id_ + 1)]['fid'] == 0:
        #         anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)

        #     else:
        #         anno_ids = self.coco.getAnnIds(imgIds=[int(id_ + 1)], iscrowd=False)

        annotations = self.coco.loadAnns(anno_ids)

        objs = []
        for obj in annotations:
            x1 = np.max((0, obj["bbox"][0]))
            y1 = np.max((0, obj["bbox"][1]))
            x2 = np.min((width-1, x1 + np.max((0, obj["bbox"][2]))))
            y2 = np.min((height-1, y1 + np.max((0, obj["bbox"][3]))))
            if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                obj["clean_bbox"] = [x1, y1, x2, y2]
                objs.append(obj)

        num_objs = len(objs)


        res = np.zeros((num_objs, 5))

        for ix, obj in enumerate(objs):
            cls = self.class_ids.index(obj["category_id"])
            res[ix, 0:4] = obj["clean_bbox"]
            res[ix, 4] = cls

        r = min(self.img_size[0] / height, self.img_size[1] / width)
        res[:, :4] *= r

        img_info = (height, width)
        resized_info = (int(height * r), int(width * r))

        # file_name = os.path.join(self.data_dir, 'Argoverse-1.1', 'tracking', self.seq_dirs[im_sid], im_name)
        # support_file_name = os.path.join(self.data_dir, 'Argoverse-1.1', 'tracking', self.seq_dirs[im_sid_support], im_name_support)

        #################support  annotation#############
        if not self.origin_support_step: # 使用当前帧作为support frame，TAL 的weight会受self.prediction_step的影响
            support_anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
            support_annotations = self.coco.loadAnns(support_anno_ids)
        else:
            support_img_id = max(target_img_id - 1, id_)
            support_anno_ids = self.coco.getAnnIds(imgIds=[support_img_id], iscrowd=False)
            support_annotations = self.coco.loadAnns(support_anno_ids)

        support_objs = []
        for obj1 in support_annotations:
            x1 = np.max((0, obj1["bbox"][0]))
            y1 = np.max((0, obj1["bbox"][1]))
            x2 = np.min((width-1, x1 + np.max((0, obj1["bbox"][2]))))
            y2 = np.min((height-1, y1 + np.max((0, obj1["bbox"][3]))))
            if obj1["area"] > 0 and x2 >= x1 and y2 >= y1:
                obj1["clean_bbox"] = [x1, y1, x2, y2]
                support_objs.append(obj1)

        support_num_objs = len(support_objs)


        support_res = np.zeros((support_num_objs, 5))

        for ix, obj1 in enumerate(support_objs):
            support_cls = self.class_ids.index(obj1["category_id"])
            support_res[ix, 0:4] = obj1["clean_bbox"]
            support_res[ix, 4] = support_cls

        support_r = min(self.img_size[0] / height, self.img_size[1] / width)
        support_res[:, :4] *= support_r

        # support_file_name = os.path.join(self.data_dir, 'Argoverse-1.1', 'tracking', self.seq_dirs[im_sid_support], im_name_support)

        # return (res, support_res, img_info, resized_info, file_name, support_file_name)
        return (res, support_res, img_info, resized_info, short_im_annos, long_im_annos, short_file_names, long_file_names)

    def load_short_imgs(self, index):
        imgs = []
        short_file_names = self.annotations[index][6]
        # short
        for i in range(self.short_cfg["frame_num"]):
            img_file = short_file_names[f'im_anno_{i*self.short_cfg["delta"]}']
            img = cv2.imread(img_file)
            imgs.append(img)
        # resize
        for i in range(self.short_cfg["frame_num"]):
            r = min(self.img_size[0] / imgs[i].shape[0], self.img_size[1] / imgs[i].shape[1])
            imgs[i] = cv2.resize(
                imgs[i],
                (int(imgs[i].shape[1] * r), int(imgs[i].shape[0] * r)),
                interpolation=cv2.INTER_LINEAR,
            ).astype(np.uint8)
        return imgs

    def load_long_imgs(self, index):
        imgs = []
        long_file_names = self.annotations[index][7]
        # long
        for i in range(self.long_cfg["frame_num"]):
            cur_id = i if self.long_cfg["include_current_frame"] else i+1
            img_file = long_file_names[f'im_anno_{cur_id*self.long_cfg["delta"]}']
            img = cv2.imread(img_file)
            imgs.append(img)
        # resize
        for i in range(self.long_cfg["frame_num"]):
            r = min(self.img_size[0] / imgs[i].shape[0], self.img_size[1] / imgs[i].shape[1])
            imgs[i] = cv2.resize(
                imgs[i],
                (int(imgs[i].shape[1] * r), int(imgs[i].shape[0] * r)),
                interpolation=cv2.INTER_LINEAR,
            ).astype(np.uint8)
        return imgs

    def pull_item(self, index):
        id_ = self.ids[index]

        res, support_res, img_info, resized_info, _, _, _, _ = self.annotations[index]

        # img = self.load_resized_img(index)
        # support_img = self.load_support_resized_img(index)
        short_imgs = self.load_short_imgs(index)
        long_imgs = self.load_long_imgs(index)

        return short_imgs, long_imgs, res.copy(), support_res.copy(), img_info, np.array([id_])

    @Dataset.mosaic_getitem
    def __getitem__(self, index):
        """
        One image / label pair for the given index is picked up \
        and pre-processed.
        Args:
            index (int): data index
        Returns:
            img (numpy.ndarray): pre-processed image
            padded_labels (torch.Tensor): pre-processed label data. \
                The shape is :math:`[self.max_labels, 5]`. \
                each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
            info_img : tuple of h, w, nh, nw, dx, dy.
                h, w (int): original shape of the image
                nh, nw (int): shape of the resized image without padding
                dx, dy (int): pad size
            id_ (int): same as the input index. Used for evaluation.
        """
        short_imgs, long_imgs, target, support_target, img_info, img_id = self.pull_item(index)

        if self.preproc is not None:

            short_imgs, long_imgs, target, support_target = self.preproc(short_imgs, long_imgs, (target, support_target), self.input_dim)

        # return np.concatenate((img, support_img), axis=0), (target, support_target), img_info, img_id
        if len(long_imgs) > 0:
            return np.concatenate(short_imgs, axis=0), np.concatenate(long_imgs, axis=0), (target, support_target), img_info, img_id
        else: # 不使用long支路的情况
            return np.concatenate(short_imgs, axis=0), np.zeros((0, )), (target, support_target), img_info, img_id
