import sys
import numpy as np
import pandas as pd
from math import sqrt
from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage import measure
from skimage.measure import label, regionprops, regionprops_table
import timeit

import matplotlib.pyplot as plt


# image = data.hubble_deep_field()[0:500, 0:500]
image = np.load(sys.argv[1]).astype(np.uint8)
# print(np.unique(image))

# DOWNSAMPLE_SCALE = 4

# image = image[
#     ::DOWNSAMPLE_SCALE, 
#     ::DOWNSAMPLE_SCALE, 
#     ::DOWNSAMPLE_SCALE
# ]


label_img = label(image)
regions = regionprops(label_img)
# print(len(regions))

props_dict = regionprops_table(
    label_img,
    properties=('centroid', 'area'),
)

print(pd.DataFrame(props_dict))
# labels = measure.label(image)

# print(f"\n\n3d array with connected blobs labelled of type {type(labels)}:")
# # print(labels)



# def extract_blobs_from_labelled_array(labelled_array):
#     # The goal is to obtain lists of the coordinates
#     # Of each distinct blob.

#     blobs = []

#     label = 1
#     while True:
#         indices_of_label = np.where(labelled_array==label)
#         if not indices_of_label[0].size > 0:
#             break
#         else:
#             blob =list(zip(*indices_of_label))
#             label+=1
#             blobs.append(blob)
            
#     return blobs


# blobs = extract_blobs_from_labelled_array(labels)

# print(blobs)









# blobs = blob_log(image, max_sigma=100)
# print(blobs)
############

# blobs_log = blob_log(image_gray, max_sigma=30, num_sigma=10, threshold=0.1)

# # Compute radii in the 3rd column.
# blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

# blobs_dog = blob_dog(image_gray, max_sigma=30, threshold=0.1)
# blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)

# blobs_doh = blob_doh(image_gray, max_sigma=30, threshold=0.01)

# blobs_list = [blobs_log, blobs_dog, blobs_doh]
# colors = ['yellow', 'lime', 'red']
# titles = ['Laplacian of Gaussian', 'Difference of Gaussian', 'Determinant of Hessian']
# sequence = zip(blobs_list, colors, titles)

# fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharex=True, sharey=True)
# ax = axes.ravel()

# for idx, (blobs, color, title) in enumerate(sequence):
#     ax[idx].set_title(title)
#     ax[idx].imshow(image)
#     for blob in blobs:
#         y, x, r = blob
#         c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
#         ax[idx].add_patch(c)
#     ax[idx].set_axis_off()

# plt.tight_layout()
# plt.show()