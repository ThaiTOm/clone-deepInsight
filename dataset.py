import numbers
import os
import queue as Queue
import threading
from typing import Iterable

import numpy as np
import torch
from functools import partial
from torch import distributed
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from utils.utils_distributed_sampler import DistributedSampler
from utils.utils_distributed_sampler import get_dist_info, worker_init_fn
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
from PIL import Image
import cv2

def iccFileFixed(image_path):
    img = Image.open(image_path)
    img.info.pop('icc_profile', None)
    img.save(image_path)

class CustomDataset(Dataset):
    def __init__(self, data_file, transform):
        self.images_dir = []
        self.labels = []
        self.transform = transform

        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                image_dir, label = line.strip().split('\t')

                image_dir = image_dir.replace("/kaggle", "kaggle")
                if "dataCollection" not in image_dir:
                    self.images_dir.append(image_dir)
                    self.labels.append(int(label))
        label = len(set(self.labels))
        for idx, folder_name in enumerate(os.listdir("kaggle/input/dataCollection/dataCollection/")):
            folder_name = f"kaggle/input/dataCollection/dataCollection/{folder_name}"
            ext = os.listdir(folder_name)[0].split(".")[-1]
            file_name = "augmented_image"
            for i in range(1, 51):
                file_ = f"{folder_name}/{file_name}_{i}.{ext}"

                self.images_dir.append(file_)
                self.labels.append(int(label + idx))

        # Combine and shuffle the data
        combined = list(zip(self.images_dir, self.labels))
        random.shuffle(combined)
        self.images_dir, self.labels = zip(*combined)

        # Convert back to lists if needed
        self.images_dir = list(self.images_dir)
        self.labels = list(self.labels)

        print(f"Loaded {len(self.images_dir)} images with {len(set(self.labels))} unique labels.")

    def __len__(self):
        return len(self.images_dir)

    def __getitem__(self, idx):
        img_path = self.images_dir[idx]
        label = self.labels[idx]

        if not os.path.isdir(img_path) and os.path.exists(img_path):
            try:
                iccFileFixed(img_path)
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            except Exception as e:
                raise FileNotFoundError(
                    f"Error opening image: {img_path}. {str(e)}, We got {os.path.isdir(img_path)} is dir")

            if self.transform:
                image = self.transform(image=image)["image"]
        else:
            img_path = os.path.join(img_path, os.listdir(img_path)[0])
            try:
                iccFileFixed(img_path)
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            except Exception as e:
                print(FileNotFoundError(
                    f"Error opening image: {img_path}. {str(e)}. The default path is {self.images_dir[idx]}, we got {os.listdir(img_path)}"))
                img_path = "kaggle/input/myfolder/content/all/train/class1681/qr125_jpg.rf.68be0b0e1167e631aa7b6f830f1de1ca33.jpg"
                label = 1
                iccFileFixed(img_path)
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if self.transform:
                image = self.transform(image=image)["image"]

        return image, label


def get_dataloader(
        root_dir,
        local_rank,
        batch_size,
        dali=False,
        dali_aug=False,
        seed=2048,
        num_workers=2,
) -> Iterable:
    blur = A.AdvancedBlur(
        blur_limit=(3, 15),  # ScaleIntType
        sigma_x_limit=(0.2, 1.0),  # ScaleFloatType
        sigma_y_limit=(0.2, 1.0),  # ScaleFloatType
        sigmaX_limit=None,  # ScaleFloatType | None
        sigmaY_limit=None,  # ScaleFloatType | None
        rotate_limit=90,  # ScaleIntType
        beta_limit=(0.5, 8.0),  # ScaleFloatType
        noise_limit=(0.9, 1.1),  # ScaleFloatType
        always_apply=None,  # bool | None
        p=1.0,  # float
    )

    affine = A.Affine(
        scale=None,  # ScaleFloatType | dict[str, Any] | None
        translate_percent=None,  # ScaleFloatType | dict[str, Any] | None
        translate_px=None,  # ScaleIntType | dict[str, Any] | None
        rotate=None,  # ScaleFloatType | None
        shear=None,  # ScaleFloatType | dict[str, Any] | None
        interpolation=1,  # <class 'int'>
        mask_interpolation=0,  # int
        cval=0,  # ColorType
        cval_mask=0,  # ColorType
        mode=0,  # int
        fit_output=False,  # bool
        keep_ratio=False,  # bool
        rotate_method="largest_box",  # Literal['largest_box', 'ellipse']
        balanced_scale=False,  # bool
        always_apply=None,  # bool | None
        p=1.0,  # float
    )
    CLAHE = A.CLAHE(
        clip_limit=4.0,  # ScaleFloatType
        tile_grid_size=(8, 8),  # tuple[int, int]
        always_apply=None,  # bool | None
        p=1.0,  # float
    )

    elasticTransform = A.ElasticTransform(
        alpha=1500,  # float
        sigma=50,  # float
        alpha_affine=None,  # None
        interpolation=1,  # <class 'int'>
        border_mode=4,  # int
        value=None,  # ScalarType | list[ScalarType] | None
        mask_value=None,  # ScalarType | list[ScalarType] | None
        always_apply=None,  # bool | None
        approximate=False,  # bool
        same_dxdy=False,  # bool
        p=1.0,  # float
    )
    noise = A.GaussNoise(
        var_limit=(10.0, 150.0),  # ScaleFloatType
        mean=0,  # float
        per_channel=True,  # bool
        noise_scale_factor=1,  # float
        always_apply=None,  # bool | None
        p=1.0,  # float
    )

    compression = A.ImageCompression(
        quality_lower=None,  # int | None
        quality_upper=None,  # int | None
        compression_type=0,  # ImageCompressionType
        quality_range=(10, 100),  # tuple[int, int]
        always_apply=None,  # bool | None
        p=1.0,  # float
    )

    sharpen = A.Sharpen(
        alpha=(0.2, 0.5),  # tuple[float, float]
        lightness=(0.5, 1.0),  # tuple[float, float]
        always_apply=None,  # bool | None
        p=1.0,  # float
    )

    ringingShoot = A.RingingOvershoot(
        blur_limit=(2, 15),  # ScaleIntType
        cutoff=(0.7853981633974483, 1.5707963267948966),  # ScaleFloatType
        always_apply=None,  # bool | None
        p=0.5,  # float
    )

    sunFlare = A.RandomSunFlare(
        flare_roi=(0, 0, 1, 0.5),  # tuple[float, float, float, float]
        angle_lower=None,  # float | None
        angle_upper=None,  # float | None
        num_flare_circles_lower=None,  # int | None
        num_flare_circles_upper=None,  # int | None
        src_radius=250,  # int
        src_color=(255, 255, 255),  # tuple[int, ...]
        angle_range=(0, 1),  # tuple[float, float]
        num_flare_circles_range=(6, 10),  # tuple[int, int]
        always_apply=None,  # bool | None
        p=0.8,  # float
    )

    randomRain = A.RandomRain(
        slant_lower=None,  # int | None
        slant_upper=None,  # int | None
        slant_range=(-5, 5),  # tuple[int, int]
        drop_length=10,  # int
        drop_width=1,  # int
        drop_color=(200, 200, 200),  # tuple[int, int, int]
        blur_value=4,  # int
        brightness_coefficient=0.7,  # float
        rain_type=None,  # RainMode | None
        always_apply=None,  # bool | None
        p=0.2,  # float
    )

    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.Rotate(limit=180, p=1),
        blur,
        CLAHE,
        affine,
        elasticTransform,
        noise,
        ringingShoot,
        sunFlare,
        compression,
        sharpen,
        randomRain,

        A.LongestMaxSize(max_size=224, interpolation=3),
        A.PadIfNeeded(min_height=224, min_width=224, border_mode=0, value=(0, 0, 0)),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    train_set = CustomDataset(data_file='kaggle/input/product-vietnamese/data.txt', transform=transform)

    rank, world_size = get_dist_info()
    train_sampler = DistributedSampler(
        train_set, num_replicas=world_size, rank=rank, shuffle=True, seed=seed)

    if seed is None:
        init_fn = None
    else:
        init_fn = partial(worker_init_fn, num_workers=num_workers, rank=rank, seed=seed)

    train_loader = DataLoaderX(
        local_rank=local_rank,
        dataset=train_set,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=init_fn,
    )

    return train_loader


class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, local_rank, max_prefetch=6):
        super(BackgroundGenerator, self).__init__()
        self.queue = Queue.Queue(max_prefetch)
        self.generator = generator
        self.local_rank = local_rank
        self.daemon = True
        self.start()

    def run(self):
        torch.cuda.set_device(self.local_rank)
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


class DataLoaderX(DataLoader):

    def __init__(self, local_rank, **kwargs):
        super(DataLoaderX, self).__init__(**kwargs)
        self.stream = torch.cuda.Stream(local_rank)
        self.local_rank = local_rank

    def __iter__(self):
        self.iter = super(DataLoaderX, self).__iter__()
        self.iter = BackgroundGenerator(self.iter, self.local_rank)
        self.preload()
        return self

    def preload(self):
        self.batch = next(self.iter, None)
        if self.batch is None:
            return None
        with torch.cuda.stream(self.stream):
            for k in range(len(self.batch)):
                self.batch[k] = self.batch[k].to(device=self.local_rank, non_blocking=True)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch


class MXFaceDataset(Dataset):
    def __init__(self, root_dir, local_rank):
        super(MXFaceDataset, self).__init__()
        self.transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
             ])
        self.root_dir = root_dir
        self.local_rank = local_rank
        path_imgrec = os.path.join(root_dir, 'train.rec')
        path_imgidx = os.path.join(root_dir, 'train.idx')
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = np.array(range(1, int(header.label[0])))
        else:
            self.imgidx = np.array(list(self.imgrec.keys))

    def __getitem__(self, index):
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = torch.tensor(label, dtype=torch.long)
        sample = mx.image.imdecode(img).asnumpy()
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label

    def __len__(self):
        return len(self.imgidx)


class SyntheticDataset(Dataset):
    def __init__(self):
        super(SyntheticDataset, self).__init__()
        img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.int32)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).squeeze(0).float()
        img = ((img / 255) - 0.5) / 0.5
        self.img = img
        self.label = 1

    def __getitem__(self, index):
        return self.img, self.label

    def __len__(self):
        return 1000000


def dali_data_iter(
        batch_size: int, rec_file: str, idx_file: str, num_threads: int,
        initial_fill=32768, random_shuffle=True,
        prefetch_queue_depth=1, local_rank=0, name="reader",
        mean=(127.5, 127.5, 127.5),
        std=(127.5, 127.5, 127.5),
        dali_aug=False
):
    """
    Parameters:
    ----------
    initial_fill: int
        Size of the buffer that is used for shuffling. If random_shuffle is False, this parameter is ignored.

    """
    rank: int = distributed.get_rank()
    world_size: int = distributed.get_world_size()
    import nvidia.dali.fn as fn
    import nvidia.dali.types as types
    from nvidia.dali.pipeline import Pipeline
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator

    def dali_random_resize(img, resize_size, image_size=112):
        img = fn.resize(img, resize_x=resize_size, resize_y=resize_size)
        img = fn.resize(img, size=(image_size, image_size))
        return img

    def dali_random_gaussian_blur(img, window_size):
        img = fn.gaussian_blur(img, window_size=window_size * 2 + 1)
        return img

    def dali_random_gray(img, prob_gray):
        saturate = fn.random.coin_flip(probability=1 - prob_gray)
        saturate = fn.cast(saturate, dtype=types.FLOAT)
        img = fn.hsv(img, saturation=saturate)
        return img

    def dali_random_hsv(img, hue, saturation):
        img = fn.hsv(img, hue=hue, saturation=saturation)
        return img

    def multiplexing(condition, true_case, false_case):
        neg_condition = condition ^ True
        return condition * true_case + neg_condition * false_case

    condition_resize = fn.random.coin_flip(probability=0.1)
    size_resize = fn.random.uniform(range=(int(112 * 0.5), int(112 * 0.8)), dtype=types.FLOAT)
    condition_blur = fn.random.coin_flip(probability=0.2)
    window_size_blur = fn.random.uniform(range=(1, 2), dtype=types.INT32)
    condition_flip = fn.random.coin_flip(probability=0.5)
    condition_hsv = fn.random.coin_flip(probability=0.2)
    hsv_hue = fn.random.uniform(range=(0., 20.), dtype=types.FLOAT)
    hsv_saturation = fn.random.uniform(range=(1., 1.2), dtype=types.FLOAT)

    pipe = Pipeline(
        batch_size=batch_size, num_threads=num_threads,
        device_id=local_rank, prefetch_queue_depth=prefetch_queue_depth, )
    condition_flip = fn.random.coin_flip(probability=0.5)
    with pipe:
        jpegs, labels = fn.readers.mxnet(
            path=rec_file, index_path=idx_file, initial_fill=initial_fill,
            num_shards=world_size, shard_id=rank,
            random_shuffle=random_shuffle, pad_last_batch=False, name=name)
        images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)
        if dali_aug:
            images = fn.cast(images, dtype=types.UINT8)
            images = multiplexing(condition_resize, dali_random_resize(images, size_resize, image_size=112), images)
            images = multiplexing(condition_blur, dali_random_gaussian_blur(images, window_size_blur), images)
            images = multiplexing(condition_hsv, dali_random_hsv(images, hsv_hue, hsv_saturation), images)
            images = dali_random_gray(images, 0.1)

        images = fn.crop_mirror_normalize(
            images, dtype=types.FLOAT, mean=mean, std=std, mirror=condition_flip)
        pipe.set_outputs(images, labels)
    pipe.build()
    return DALIWarper(DALIClassificationIterator(pipelines=[pipe], reader_name=name, ))


@torch.no_grad()
class DALIWarper(object):
    def __init__(self, dali_iter):
        self.iter = dali_iter

    def __next__(self):
        data_dict = self.iter.__next__()[0]
        tensor_data = data_dict['data'].cuda()
        tensor_label: torch.Tensor = data_dict['label'].cuda().long()
        tensor_label.squeeze_()
        return tensor_data, tensor_label

    def __iter__(self):
        return self

    def reset(self):
        self.iter.reset()
