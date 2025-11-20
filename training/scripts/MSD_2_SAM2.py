#!/usr/bin/env python3
# msd_to_sam2.py

import os
import argparse
import nibabel as nib
import numpy as np
from PIL import Image
from tqdm import tqdm
import json


def normalize_to_uint8(volume):
    """
    将医学图像归一化到 [0, 255] 并转为 uint8
    注意：不同 MSD 任务的 intensity range 不同，可根据需要调整
    """
    volume = volume.astype(np.float32)
    # 简单 min-max 归一化（可替换为窗宽窗位）
    p1, p99 = np.percentile(volume, [1, 99])
    volume = np.clip(volume, p1, p99)
    volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
    return (volume * 255).astype(np.uint8)


def convert_nii_to_png_slices(
    image_path: str,
    label_path: str,
    output_dir: str,
    patient_id: str,
    axis: int = 2,  # 默认沿 Z 轴切片（axial view）
    skip_empty: bool = False,
):
    """
    将一个患者的 3D NIfTI 图像+标签转换为 PNG 序列
    """
    # 加载图像和标签
    img_nii = nib.load(image_path)
    lbl_nii = nib.load(label_path)

    img_data = img_nii.get_fdata()
    lbl_data = lbl_nii.get_fdata()

    # 沿指定轴切片（默认Z轴）
    if axis == 0:
        slices = [(img_data[i], lbl_data[i]) for i in range(img_data.shape[0])]
    elif axis == 1:
        slices = [(img_data[:, i, :], lbl_data[:, i, :]) for i in range(img_data.shape[1])]
    elif axis == 2:
        slices = [(img_data[:, :, i], lbl_data[:, :, i]) for i in range(img_data.shape[2])]
    else:
        raise ValueError("axis must be 0, 1, or 2")

    # 创建输出目录
    patient_dir = os.path.join(output_dir, f"JPEGimages/video_{patient_id}/")
    os.makedirs(patient_dir, exist_ok=True)
    mask_dir = os.path.join(output_dir, f"Annotations/video_{patient_id}/")
    os.makedirs(mask_dir, exist_ok=True)

    valid_frame_count = 0
    for i, (img_slice, mask_slice) in enumerate(slices):
        # 只保留有肿瘤/器官的切片（可选）
        if skip_empty and np.sum(mask_slice > 0) == 0:
            continue  # 跳过无标注切片（节省空间）

        # 图像归一化并保存
        img_normalized = normalize_to_uint8(img_slice)
        img_pil = Image.fromarray(img_normalized, mode='L')
        img_pil.save(os.path.join(patient_dir, f"{i:05d}.jpg"))

        # 掩码处理：二值化（SAM2 需要 0/255 或 0/1）
        mask_binary = (mask_slice > 0).astype(np.uint8) * 255
        mask_pil = Image.fromarray(mask_binary, mode='L')
        mask_pil.save(os.path.join(mask_dir, f"{i:05d}.png"))

        valid_frame_count += 1

    return valid_frame_count > 0  # 是否至少有一个有效帧


def main():
    parser = argparse.ArgumentParser(description="Convert MSD dataset to SAM2-compatible format")
    parser.add_argument("--msd_root", type=str, required=True, help="Path to MSD dataset root (e.g., Task07_BrainTumour)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for SAM2 format")
    parser.add_argument("--axis", type=int, default=2, choices=[0,1,2], help="Axis to slice along (0=sagittal, 1=coronal, 2=axial)")
    parser.add_argument("--skip_empty", action="store_true", help="Skip slices with no annotation (recommended)")
    args = parser.parse_args()

    images_dir = os.path.join(args.msd_root, "imagesTr")
    labels_dir = os.path.join(args.msd_root, "labelsTr")

    assert os.path.exists(images_dir), f"Images dir not found: {images_dir}"
    assert os.path.exists(labels_dir), f"Labels dir not found: {labels_dir}"

    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(".nii.gz")])
    label_files = sorted([f for f in os.listdir(labels_dir) if f.endswith(".nii.gz")])

    assert len(image_files) == len(label_files), "Image and label counts mismatch!"

    os.makedirs(args.output_dir, exist_ok=True)
    valid_patients = []

    print(f"Converting {len(image_files)} patients from MSD to SAM2 format...")

    for img_file, lbl_file in tqdm(zip(image_files, label_files), total=len(image_files)):
        patient_id = img_file.replace(".nii.gz", "")
        img_path = os.path.join(images_dir, img_file)
        lbl_path = os.path.join(labels_dir, lbl_file)

        try:
            has_valid = convert_nii_to_png_slices(
                img_path, lbl_path, args.output_dir, patient_id, axis=args.axis, skip_empty = args.skip_empty
            )
            if has_valid:
                valid_patients.append(f"video_{patient_id}")
        except Exception as e:
            print(f"Error processing {patient_id}: {e}")

    # 生成 filelist.txt（供 SAM2 训练使用）
    filelist_path = os.path.join(args.output_dir, "filelist.txt")
    with open(filelist_path, "w") as f:
        for pid in valid_patients:
            f.write(pid + "\n")

    print(f"\n Conversion complete!")
    print(f" - Output dir: {args.output_dir}")
    print(f" - Valid patients: {len(valid_patients)}")
    print(f" - File list saved to: {filelist_path}")


if __name__ == "__main__":
    main()