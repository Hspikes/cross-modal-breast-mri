path = "./data/raw/t1c"
origin_path = "./data/raw/b800"
segment_ouput_folder = "./data/raw_match/segment"
c_output_folder = "./data/raw_match/t1c"
b800_output_folder = "./data/raw_match/b800"
import os

import SimpleITK as sitk


def Rigid_registration(fixed_image, moving_image):

    initial_transform = sitk.CenteredTransformInitializer(
        fixed_image,
        moving_image,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    registration_method = sitk.ImageRegistrationMethod()

    # Similarity metric settings.
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=400)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.1)

    registration_method.SetInterpolator(sitk.sitkLinear)

    # Optimizer settings.
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=100,
        convergenceMinimumValue=1e-5,
        convergenceWindowSize=10,
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Don't optimize in-place, we would possibly like to run this cell multiple times.
    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    final_transform = registration_method.Execute(
        sitk.Cast(fixed_image, sitk.sitkFloat32),
        sitk.Cast(moving_image, sitk.sitkFloat32),
    )

    moving_resampled = sitk.Resample(
        moving_image,
        moving_image,
        final_transform,
        sitk.sitkLinear,
        float(0),
        moving_image.GetPixelID(),
    )

    return moving_resampled


def resample_binary_image(image, new_spacing):

    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    new_size = [
        int(round(osz * ospc / nspc))
        for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)

    resampled_image = resampler.Execute(image)
    return resampled_image


def resample_image(image, new_spacing):
    """
    对输入图像进行重采样。

    参数：
        image (SimpleITK.Image): 需要被重采样的输入图像。
        new_spacing (tuple): 新的像素间距 (深度，高度，宽度)。

    返回：
        SimpleITK.Image: 重采样后的图像。
    """
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    new_size = [
        int(round(osz * ospc / nspc))
        for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetInterpolator(sitk.sitkLinear)

    resampled_image = resampler.Execute(image)
    return resampled_image


def expand_square_bounding_box(segmentation_image, target_size=64):

    # 使用LabelShapeStatisticsImageFilter获取连通区域的统计信息
    label_statistics = sitk.LabelShapeStatisticsImageFilter()
    label_statistics.Execute(segmentation_image)

    # 获取前景（标签为1）的包围盒
    bounding_box = label_statistics.GetBoundingBox(1)

    # 获取最小包围盒的尺寸
    size = bounding_box[3:5]
    max_size = max(size[:2])  # 只保证XY方向是正方形，取其中较大的尺寸

    # 计算外扩后的边长
    target_size = max(target_size, max_size)

    # 计算额外的边长
    extra_size = (target_size - max_size) // 2

    # 计算新的起始点
    start_point = [
        bounding_box[0] - extra_size,
        bounding_box[1] - extra_size,
        bounding_box[2],
    ]

    # 构建新的包围盒信息 [xmin, ymin, zmin, xsize, ysize, zsize]
    expanded_box = start_point + [target_size, target_size, bounding_box[5]]

    return expanded_box


def center_crop_box(image, target_size=256):
    label_statistics = sitk.LabelShapeStatisticsImageFilter()
    label_statistics.Execute(image)
    X, Y, _ = image.GetSize()
    # 获取前景（标签为1）的包围盒
    bounding_box = label_statistics.GetBoundingBox(1)
    center_point = [X // 2, Y // 2]
    len_size = target_size // 2
    center_box = [
        center_point[0] - len_size,
        center_point[1] - len_size,
        bounding_box[2],
        target_size,
        target_size,
        bounding_box[5],
    ]

    return center_box


def crop_to_bounding_box(image, bounding_box):
    # 获取最小包围盒的起始点和尺寸
    start_point = bounding_box[:3]
    size = bounding_box[3:6]

    # 获取图像的大小
    image_size = image.GetSize()
    # 调整裁剪区域，确保不超出图像范围
    for i in range(3):
        if start_point[i] < 0:
            return None
            # size[i] += start_point[i]
            # start_point[i] = 0
        if start_point[i] + size[i] > image_size[i]:
            return None
            # size[i] = image_size[i] - start_point[i]
    # 使用SimpleITK的裁剪功能对图像进行裁剪
    cropped_image = sitk.RegionOfInterest(image, size, start_point)
    return cropped_image


def main():
    for data in os.listdir(path):
        t1c_file_path = os.path.join(path, data)
        b800_file_path = t1c_file_path.replace("t1c", "b800")
        print(b800_file_path)
        t1c_mask_file_path = t1c_file_path.replace("ORIGIN", "MASK")
        b800_mask_file_path = b800_file_path.replace("ORIGIN", "MASK")
        # 读取分割图像及各个序列图像
        t1c_segment_image = sitk.ReadImage(t1c_mask_file_path)
        b800_segment_image = sitk.ReadImage(b800_mask_file_path)

        b800_image = sitk.ReadImage(b800_file_path, sitk.sitkFloat32)
        t1c_image = sitk.ReadImage(t1c_file_path, sitk.sitkFloat32)
        # 对齐
        b800_image = Rigid_registration(t1c_image, b800_image)
        # 重采样成1*1*1
        new_spacing = (1.0, 1.0, 1.0)
        resample_t1c_segment_image = resample_binary_image(
            t1c_segment_image, new_spacing
        )
        resample_b800_segment_image = resample_binary_image(
            b800_segment_image, new_spacing
        )

        resample_c_image = resample_image(t1c_image, new_spacing)
        resample_b800_image = resample_image(b800_image, new_spacing)

        c_regis_image = resample_c_image
        c_regis_segment_image = resample_t1c_segment_image

        c_bounding_box = center_crop_box(c_regis_segment_image)
        dwi_bounding_box = center_crop_box(resample_b800_segment_image)
        print(dwi_bounding_box)
        min_size = min(c_bounding_box[-1], dwi_bounding_box[-1])
        c_bounding_box[-1] = min_size
        dwi_bounding_box[-1] = min_size
        # 裁剪分割图像
        cropped_c_segmentation = crop_to_bounding_box(
            c_regis_segment_image, c_bounding_box
        )
        cropped_dwi_segmentation = crop_to_bounding_box(
            resample_b800_segment_image, dwi_bounding_box
        )
        if cropped_c_segmentation == None or cropped_dwi_segmentation == None:
            print(data)
            continue

        # 裁剪原始图像
        cropped_c = crop_to_bounding_box(c_regis_image, c_bounding_box)
        cropped_b800 = crop_to_bounding_box(resample_b800_image, dwi_bounding_box)

        # 生成保存路径和文件名
        data_num = data.split("_")[0]
        save_path_segmentation = os.path.join(
            segment_ouput_folder, f"{data_num}.nii.gz"
        )
        save_path_t1c = os.path.join(c_output_folder, f"{data_num}_t1c.nii.gz")
        save_path_b800 = os.path.join(b800_output_folder, f"{data_num}_b800.nii.gz")

        # 保存裁剪后的分割图像
        sitk.WriteImage(cropped_dwi_segmentation, save_path_segmentation)

        # 保存裁剪后的原始图像
        sitk.WriteImage(cropped_c, save_path_t1c)
        try:
            sitk.WriteImage(cropped_b800, save_path_b800)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Failed to write registered DWI volume: {save_path_b800}"
            ) from exc


if __name__ == "__main__":
    if not os.path.exists(segment_ouput_folder):
        os.makedirs(segment_ouput_folder)
    if not os.path.exists(b800_output_folder):
        os.makedirs(b800_output_folder)
    if not os.path.exists(c_output_folder):
        os.makedirs(c_output_folder)
    main()
