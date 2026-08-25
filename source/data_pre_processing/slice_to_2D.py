import os

import matplotlib.pyplot as plt
import SimpleITK as sitk

raw_dir = "./data/raw_match"
b800_dir = raw_dir + "/" + "b800"
t1c_dir = raw_dir + "/" + "t1c"
segment_dir = raw_dir + "/" + "segment"

b800_output_dir = b800_dir.replace("raw_match", "raw_png")
t1c_output_dir = t1c_dir.replace("raw_match", "raw_png")
segment_output_dir = segment_dir.replace("raw_match", "raw_png")

for in_dir, out_dir in zip(
    [b800_dir, t1c_dir, segment_dir],
    [b800_output_dir, t1c_output_dir, segment_output_dir],
):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for data in os.listdir(in_dir):
        sub_dir = data.split(".")[0]
        dir = os.path.join(out_dir, sub_dir)
        if not os.path.exists(dir):
            os.mkdir(dir)
        print(sub_dir)
        file_path = os.path.join(in_dir, data)
        image = sitk.ReadImage(file_path)
        numpy_image = sitk.GetArrayFromImage(image)
        for i in range(numpy_image.shape[0]):
            fig = numpy_image[i, :, :]
            plt.imsave(os.path.join(dir, "{}.png".format(i)), fig, cmap="gray")
            # plt.imsave('test.png',fig, cmap='gray')
