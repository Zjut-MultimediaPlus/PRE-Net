import logging
import os
import pickle
import random
import re
from datetime import datetime, timedelta

import numpy as np
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm
import xarray as xr
import torch.nn as nn

from utils import utils

css_dir_img = Path("/opt/data/private/PERSIANN-CCS/")
pdir_dir_img = Path("/opt/data/private/PDIR/")
gpm_dir_img = Path("/opt/data/private/GPM/")

def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.hdf':
        return xr.open_dataset(filename)
    elif ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)

def load_image_xr(filename):
    need_index = np.ones(281, bool)
    need_index[-40:] = False
    need_index[-170:-130] = False

    ds = xr.open_dataset(filename)
    img = ds['hy_data'].values[:, :, need_index]
    mask = ds['PRECI_cldas'].values

    #  !修改1, 返回实际长度
    real_len = len(ds['TIME_dayOfYear'].values)
    return img, mask, real_len


def unique_mask_values(idx, mask_dir, mask_suffix):

    mask_file = list(mask_dir.glob('mask'+idx[5:] + '.*'))[0]
    mask = np.asarray(load_image(mask_file))

    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


    #
    # mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    #
    # mask = np.asarray(load_image(mask_file))
    #
    # if mask.ndim == 2:
    #     return np.unique(mask)
    # elif mask.ndim == 3:
    #     mask = mask.reshape(-1, mask.shape[-1])
    #     return np.unique(mask, axis=0)
    # else:
    #     raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')

class BasicDataset(Dataset):
    def __init__(self, images_dir, scale, classes, geo_norm, geo_channel):
        self.images_dir = Path(images_dir)
        self.classes = classes
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.geo_norm = geo_norm
        self.geo_channel = geo_channel


        self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(npy_file, classes, geo_norm, geo_channel):
        data = np.load(npy_file)
        ori_label = data[198]
        # 处理标签
        if classes == 2:
            label = np.zeros(ori_label.shape)
            label[ori_label >= 0.1] = 1
        else:
            label = ori_label

        # 处理卫星数据
        img = data[:197]

        dem, lat, lon, slope, aspect = data[197], data[200], data[201], data[202], data[203]

        dem_max = geo_norm[0]
        dem_min = geo_norm[1]
        dem = (dem - dem_min) / (dem_max - dem_min)

        lat = (lat + 90) / 180
        lon = (lon + 180) / 360
        slope = slope / 90
        aspect = aspect / 360

        geo = np.array([dem, lat, lon, slope, aspect])[geo_channel[0]:geo_channel[1], :, :]

        return img, geo, label

    def __getitem__(self, idx):
        name = self.ids[idx]

        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'

        try:
            img, geo, label = self.preprocess(img_file[0], self.classes, self.geo_norm, self.geo_channel)
        except ValueError:
            logging.info(f"wrong file {img_file[0]}")

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'geo': torch.as_tensor(geo.copy()).float().contiguous(),
            'label': torch.as_tensor(label.copy()).float().contiguous()
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, scale=1):
        super().__init__(images_dir, mask_dir, scale, mask_suffix='_mask')


class IR_BasicDataset(Dataset):
    def __init__(self, images_dir, scale: float = 1.0, classes: float = 1.0, mask_ratio: float = 0.25):
        self.images_dir = Path(images_dir)
        self.classes = classes
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        self.mask_items = [utils.generate_mask((1, 1, 16, 16), mask_ratio).unsqueeze(0) for _ in range(len(self.ids))]

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')


    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(npy_file, classes):
        try:
            data = xr.load_dataset(npy_file)

            ori_label = torch.as_tensor(data.PRCP.values).float()
            # 处理标签
            if classes == 2:
                label = torch.zeros(ori_label.shape)
                label[ori_label >= 0.1] = 1
            else:
                label = ori_label
            # 处理卫星数据
            img = torch.as_tensor(data.NOMChannel.values)
            # 将 NaN 值替换为 0
            img = (img - 100) / (500 - 100)
            img[torch.isnan(img)] = 0

            return img, label
        except ValueError:
            print(npy_file)

    def update_mask_item(self, idx, new_mask_item):
        self.mask_items[idx] = new_mask_item

    def __getitem__(self, idx):
        name = self.ids[idx]
        img_file = list(self.images_dir.glob(name + '.*'))
        img, label = self.preprocess(img_file[0], self.classes)
        mask_item = self.mask_items[idx]
        return {
            'index': idx,
            'image': img.contiguous(),
            'label': label.contiguous(),
            'mask_item': mask_item.contiguous(),
        }

def collate_fn(batch):
    # 将索引和数据分开
    indices = [item['index'] for item in batch]
    images = torch.stack([item['image'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    mask_items = torch.stack([item['mask_item'].reshape(1, 16, 16) for item in batch])
    return {
        'indices': indices,
        'images': images,
        'labels': labels,
        'mask_items': mask_items
    }

class Product_Dataset(Dataset):
    def __init__(self, images_dir, scale: float = 1.0, classes: float = 1.0, mask_suffix: str = '', dataname: str = 'GPM'):
        self.images_dir = Path(images_dir)
        self.classes = classes
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.dataname = dataname
        self.ids = [splitext(file)[0] for file in listdir(images_dir) if
                    isfile(join(images_dir, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')


    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(npy_file, classes):
        try:
            data = xr.load_dataset(npy_file)

            ori_label = torch.as_tensor(data.PRCP.values).float()
            # 处理标签
            if classes == 2:
                label = torch.zeros(ori_label.shape)
                label[ori_label >= 0.1] = 1
            else:
                label = ori_label
            # 处理卫星数据
            img = torch.as_tensor(data.NOMChannel.values)
            # 将 NaN 值替换为 0
            img = (img - 100) / (500 - 100)
            img[torch.isnan(img)] = 0

            return data, img, label
        except ValueError:
            print(npy_file)

    def __getitem__(self, idx):
        name = self.ids[idx]
        img_file = list(self.images_dir.glob(name + '.*'))
        data, img, label = self.preprocess(img_file[0], self.classes)

        match = re.search(r'(\d{4}\d{2}\d{2}\d{2})', name)
        date = match.group(1)  # 2023010100

        file_path = []
        if self.dataname == 'PERSIANN-CCS':
            file_path.append(f'{css_dir_img}/PERSIANN-CCS_{date}.nc')
        elif self.dataname == 'PDIR':
            file_path.append(f'{pdir_dir_img}/PDIR_{date}.nc')
        else:
            time = datetime.strptime(date, "%Y%m%d%H")
            # 减去一个小时
            adjusted_time = time - timedelta(hours=1)
            half_idx = adjusted_time - datetime.strptime('00:00:00', '%H:%M:%S')
            half_idx = half_idx.seconds // 60
            a = str(half_idx)
            b = str(30 + half_idx)
            if len(a) == 1:
                a = '0000'
            elif len(a) == 2:
                a = '00' + a
            elif len(a) == 3:
                a = '0' + a
            if len(b) == 1:
                b = '0000'
            elif len(b) == 2:
                b = '00' + b
            elif len(b) == 3:
                b = '0' + b

            # 将 datetime 对象转换回字符串
            adjusted_time = adjusted_time.strftime("%Y%m%d%H")
            file_path.append(os.path.join(gpm_dir_img, f'3B-HHR.MS.MRG.3IMERG.{adjusted_time[:-2]}-S{adjusted_time[-2:]}3000-E{adjusted_time[-2:]}5959.{b}.V07B.HDF5.nc4'))
            file_path.append(os.path.join(gpm_dir_img, f'3B-HHR.MS.MRG.3IMERG.{adjusted_time[:-2]}-S{adjusted_time[-2:]}0000-E{adjusted_time[-2:]}2959.{a}.V07B.HDF5.nc4'))

        if len(file_path) < 1:
            raise RuntimeError(f'No input file found for {adjusted_time}')

        pred = xr.load_dataset(file_path[0])
        pred = pred.sel(lat=data.lat, lon=data.lon)
        pred = pred.__xarray_dataarray_variable__.values
        pred = torch.as_tensor(pred.copy()).float().contiguous()

        if self.dataname == 'IMERG':
            p = xr.load_dataset(file_path[1])
            p = p.sel(lat=data.lat, lon=data.lon)
            p = p.__xarray_dataarray_variable__.values
            p = torch.as_tensor(p.copy()).float().contiguous()
            pred = (pred + p) / 2

        # img = self.img[idx]
        # mask = self.mask[idx]
        return {
            'image': img.contiguous(),
            'label': label.contiguous(),
            'pred': pred.contiguous()
        }


