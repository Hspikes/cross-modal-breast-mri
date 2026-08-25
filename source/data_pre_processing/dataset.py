import os

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.dataloader import default_collate


def collate_dict(batch):
    images = default_collate([item["image"] for item in batch])
    gts = default_collate([item["gt"] for item in batch])
    masks = default_collate([item["mask"] for item in batch])
    return {
        "image": images,
        "gt": gts,
        "mask": masks,
    }


class MRIDataset(Dataset):
    # 根据 data_folder 以及 sequence_list_txt 拼接出 list 每个条目的目录位置，便于后续使用
    # 指定 transform 类
    def __init__(
        self,
        data_folder,
        sequence_list_txt,
        transforms=None,
        expand_train_with_aug: bool = False,
        expand_train_fixed3: bool = False,
        aug_copies: int = 1,
    ):
        self.data_folder = data_folder
        self.zero_path = []

        self.transforms = transforms
        self.expand_train_with_aug = bool(expand_train_with_aug)
        self.expand_train_fixed3 = bool(expand_train_fixed3)
        self.aug_copies = int(aug_copies)
        self.is_train_split = "train" in str(sequence_list_txt)
        self._should_reinit_norm = False
        if transforms is not None:
            self.image_norm = transforms.image_norm
            self.gt_norm = transforms.gt_norm
            self.mask_to_tensor = transforms.mask_to_tensor
            if self.is_train_split:
                try:
                    self.train_tf = transforms.train_transform
                except AttributeError:
                    self.train_tf = None
            else:
                self.train_tf = None
            self.dup_tf = (
                getattr(transforms, "duplicate_transform", None) or self.train_tf
            )

        self.image_path_list = []
        self.sequence_path = os.path.join(data_folder, sequence_list_txt)
        with open(self.sequence_path, "r") as f:
            lines = f.readlines()
            self.sequence_list = [line.strip() for line in lines]
        for idx in self.sequence_list:
            image_sequence_dir = os.path.join(self.data_folder, "b800", idx + "_b800")
            for i in os.listdir(image_sequence_dir):
                self.image_path_list.append(os.path.join(image_sequence_dir, i))

    def __len__(self):
        base_len = len(self.image_path_list)
        if self.is_train_split and self.expand_train_fixed3:
            return base_len * 3
        if (
            self.transforms is not None
            and self.is_train_split
            and self.expand_train_with_aug
            and self.aug_copies > 0
        ):
            return base_len * (1 + self.aug_copies)
        return base_len

    # 根据 image_path_list 返回处理完成后的三类图像
    def __getitem__(self, idx):
        base_len = len(self.image_path_list)
        apply_dup_aug = False
        fixed_copy_idx = 0
        if self.is_train_split and self.expand_train_fixed3:
            base_idx = idx % base_len
            fixed_copy_idx = idx // base_len  # 0: 原图 1: 翻转 2: 旋转180
            image_path = self.image_path_list[base_idx]
        elif (
            self.transforms is not None
            and self.is_train_split
            and self.expand_train_with_aug
            and self.aug_copies > 0
        ):
            base_idx = idx % base_len
            copy_idx = idx // base_len
            apply_dup_aug = copy_idx > 0
            image_path = self.image_path_list[base_idx]
        else:
            image_path = self.image_path_list[idx]

        t1c_path = image_path.replace("b800", "t1c")
        mask_path = image_path.replace("b800", "segment")
        mask_path = mask_path.replace("_segment", "")
        return self.get_sequnence_frame(
            image_path,
            t1c_path,
            mask_sequence_path=mask_path,
            apply_dup_aug=apply_dup_aug,
            fixed_copy_idx=fixed_copy_idx,
        )

    def get_sequnence_frame(
        self,
        image_sequence_path,
        gt_seqquence_path,
        mask_sequence_path,
        apply_dup_aug: bool = False,
        fixed_copy_idx: int = 0,
    ):

        image = Image.open(image_sequence_path).convert("L")
        gt = Image.open(gt_seqquence_path).convert("L")
        mask = (
            Image.open(mask_sequence_path)
            .convert("L")
            .point(lambda p: 255 if p > 127 else 0, mode="L")
        )

        # 转化为 numpy 格式
        image = np.array(image)
        if image.max() < 160:
            idx = image_sequence_path.split("/")[-2]
            idx = idx.replace("_b800", "")
            if idx not in self.zero_path:
                self.zero_path.append(idx)

        gt = np.array(gt)
        mask = np.array(mask)

        # 固定扩容为 3 倍（训练集）：先做确定性的几何变换，再走归一化/转 tensor。
        if self.is_train_split and self.expand_train_fixed3:
            # 0: 原图
            # 1: 水平翻转（对称）
            # 2: 旋转 180 度
            if fixed_copy_idx == 1:
                image = np.fliplr(image).copy()
                gt = np.fliplr(gt).copy()
                mask = np.fliplr(mask).copy()
            elif fixed_copy_idx == 2:
                image = np.rot90(image, 2).copy()
                gt = np.rot90(gt, 2).copy()
                mask = np.rot90(mask, 2).copy()

        # 对原始图像做一定的图像增强
        if self.transforms is not None:
            if not (self.is_train_split and self.expand_train_fixed3):
                if (
                    self.is_train_split
                    and self.expand_train_with_aug
                    and self.aug_copies > 0
                ):
                    # “扩容式增强”：同一张图会同时以“原图样本 + 增强样本”形式出现
                    # - 原图样本：不做增强
                    # - 增强样本：做一次必定发生的几何变换（优先用 duplicate_transform）
                    if apply_dup_aug and self.dup_tf is not None:
                        aug = self.dup_tf(image=image, gt=gt, mask=mask)
                        image = aug["image"]
                        gt = aug["gt"]
                        mask = aug["mask"]
                else:
                    # 原有行为：训练集对每张图做随机增强（可能不触发）
                    if self.train_tf is not None:
                        aug = self.train_tf(image=image, gt=gt, mask=mask)
                        image = aug["image"]
                        gt = aug["gt"]
                        mask = aug["mask"]

            image = self.image_norm(image=image)
            gt = self.gt_norm(image=gt)
            mask = self.mask_to_tensor(image=mask)

            return {
                "image": image["image"].float(),
                "gt": gt["image"].float(),
                "mask": mask["image"].float(),
            }
        else:
            # 手工转 tensor
            image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # [1,H,W]
            gt = torch.tensor(gt, dtype=torch.float32).unsqueeze(0)
            mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
            return {"image": image, "gt": gt, "mask": mask, "zero": self.zero_path}


class FixedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, num_batches):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_batches = num_batches

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        indices = indices[: self.num_batches * self.batch_size]
        batch_indices = [
            indices[i : i + self.batch_size]
            for i in range(0, len(indices), self.batch_size)
        ]
        for batch in batch_indices:
            yield batch

    def __len__(self):
        return self.num_batches


class MRItransform:
    def __init__(self):

        self.image_norm = A.Compose(
            [A.Normalize(mean=[0.5], std=[0.5], max_pixel_value=255.0), ToTensorV2()]
        )
        self.gt_norm = A.Compose(
            [A.Normalize(mean=[0.5], std=[0.5], max_pixel_value=255.0), ToTensorV2()]
        )
        self.mask_to_tensor = ToTensorV2()  # mask 不做归一化，只转 tensor

        # 训练时随机增强（可能不触发）
        self.train_transform = A.Compose(
            [
                A.HorizontalFlip(p=0.25),
                A.Rotate(limit=15, p=0.25),
            ],
            additional_targets={"gt": "image", "mask": "mask"},
        )

        # “扩容式增强”用：确保每次至少发生一次几何变换（旋转/翻转）
        self.duplicate_transform = A.Compose(
            [
                A.OneOf(
                    [
                        A.HorizontalFlip(p=1.0),
                        A.VerticalFlip(p=1.0),
                        A.RandomRotate90(p=1.0),
                        A.Rotate(limit=15, p=1.0),
                    ],
                    p=1.0,
                )
            ],
            additional_targets={"gt": "image", "mask": "mask"},
        )


if __name__ == "__main__":
    transform = None
    data_folder = "./data"
    train_dataset = MRIDataset(data_folder, "train_list.txt", transforms=transform)

    train_dataloader = DataLoader(
        train_dataset, batch_size=8, num_workers=4, shuffle=True
    )
    eval_dataset = MRIDataset(
        data_folder=data_folder, sequence_list_txt="eval_list.txt", transforms=transform
    )
    eval_loader = DataLoader(eval_dataset, batch_size=8, num_workers=4, shuffle=False)
    train_len = len(train_dataset)
    print(train_len)

    for data in train_dataloader:
        mask = data["mask"]
        for i in range(mask.shape[0]):
            single_mask = mask[i, :, :, :].cpu().squeeze().numpy()
            if single_mask.sum() > 0:
                plt.imsave("mask.png", single_mask)

        zero = data["zero"]
    with open("train_zero.txt", "w") as f:
        f.write(zero[0] + "\n")
