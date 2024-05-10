
import numpy as np
import cv2
import random
import math
random.seed(None)

from src.data.components.custom_aug.base import *

def sharp_mask(img, line_x_erosion, line_y_erosion, line_erode=True):
  threshold = cv2.threshold(cv2.cvtColor(img,  cv2.COLOR_BGR2GRAY), 200, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
  if np.average(threshold) > 0.75:
    threshold = 1 - threshold
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
  h, w = img.shape[:2]
  final = None
  if line_erode is True:
    if h > w:
        final = line_erosion(threshold, 1, line_x_erosion)
        final = line_erosion(final, 0, line_y_erosion)
    else:
        # print("Crop row first")
        final = line_erosion(threshold, 0, line_y_erosion)
        final = line_erosion(final, 1, line_x_erosion)
  else:
    final = threshold
  final = cv2.morphologyEx(final, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
  return final


def light_mask(img):
  mask = img < (20, 20, 20)
  hbg = extract_median_color_axis(img.copy(), mask, axis = 0)
  wbg = extract_median_color_axis(img.copy(), mask, axis = 1)


def rotate_img(img, angle, mask, bg_color):
  h, w = img.shape[:2]
  corner = np.zeros((4, 3))
  corner[1:3, 1] += h
  corner[2:4, 0] += w
  corner[:, 2] = 1
  # print(corner)
  rotate_matrix = cv2.getRotationMatrix2D((w / 2, h / 2), -angle, 1.)
  # print(rotate_matrix.shape)
  transformed_corner = np.vstack([np.matmul(rotate_matrix, c.T) for c in corner])
  # print(transformed_corner)
  masked_img = np.concatenate([img, mask[:, :, None].astype(np.uint8)], axis=2)
  rotate_matrix[:, 2] += [- min(transformed_corner[:, 0]), - min(transformed_corner[:, 1])]
  test = np.vstack([np.matmul(rotate_matrix, c.T) for c in corner])
  # print(test)
  w1 = max(transformed_corner[:, 0]) - min(transformed_corner[:, 0])
  h1 = max(transformed_corner[:, 1]) - min(transformed_corner[:, 1])
  warped = cv2.warpAffine(src=masked_img,
                          M=rotate_matrix,
                          dsize=(int(w1), int(h1)),
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=bg_color)
  return warped[:, :, :img.shape[2]], warped[:, :, -1]

def shear_img(img, mask, shear_x, shear_y, bg_color):
  h, w = img.shape[:2]
  masked_img = np.concatenate([img, mask[:, :, None].astype(np.uint8)], axis=2)

  shear_kernel = np.float32([[1, shear_x, max(0, - shear_x * h)],
                         [shear_y, 1, max(0, - shear_y * w)],
                         [0, 0, 1]])
  # iscale = min(abs(shear_x),abs(shear_y))
  output_size = (int(w + abs(shear_x) * h), int(h + abs(shear_y) * w))
  warped = cv2.warpPerspective(masked_img, 
                               shear_kernel, 
                               dsize = output_size, 
                               borderMode=cv2.BORDER_CONSTANT, 
                               borderValue = bg_color,
                               flags=cv2.INTER_LINEAR)
  return warped[:, :, :img.shape[2]], warped[:, :, -1]

def augment_img(img, 
                rotate=0, 
                shear_x=0, 
                shear_y=0, 
                noise=0, 
                logger = None, 
                debug = None, 
                keep_mask = True):
  # cài đặt thông số
  skew_angle = 0
  shear_x_level = 0
  shear_y_level = 0
  if len(rotate) == 2:
    skew_angle = rotate[0] + random.random() * (rotate[1] - rotate[0])
  else:
    skew_angle = rotate[0]
  
  if len(shear_x) == 2: 
    shear_x_level = shear_x[0] + random.random() * (shear_x[1] - shear_x[0])
  else:
    shear_x_level = shear_x[0]
  
  if len(shear_y) == 2: 
    shear_y_level = shear_y[0] + random.random() * (shear_y[1] - shear_y[0])
  else:
    shear_y_level = shear_y[0]
  
  log = "Rotate {0} degree, Shear_x {1}, Shear_y {2}, Noise".format(skew_angle, shear_x_level, shear_y_level)

  # trích xuất mask
  sub_img = smooth(img, 20, 50)
  thresh = np.mean(sub_img, axis=(0, 1))
  # print("Background filter: ", bg_color)
  mask = np.max((sub_img < thresh).astype(int), axis = 2).astype(np.uint8)

  # trich xuat bg
  bg_mask = img < (20, 20, 20)
  bg_color = extract_median_color_axis(img.copy(), bg_mask, axis = (0, 1)).tolist()
  bg_color.append(0)

  # kéo 
  sheared, sheared_mask = shear_img(img, mask, shear_x_level, shear_y_level, bg_color)
  # xoay
  rotated, rotated_mask = rotate_img(sheared, skew_angle, sheared_mask, bg_color)
  # cắt
  if keep_mask is True:
    merged = np.concatenate([rotated, rotated_mask[:, :, None].astype(np.uint8)], axis=2)
    output = crop_img(merged, rotated_mask, 0, 0, 0, 0, bg_color, smoothing = False)
  else:
    output = crop_img(rotated, rotated_mask, 0, 0, 0, 0, bg_color[:3], smoothing = False)
  # log anh ra ngoai
  if debug is not None:
    # print("Debug_folder: " + debug)
    mask_img = mask * 255
    # print(mask_img.shape, mask_img.dtype)
    # print(cv2.imwrite(debug.format("_original"), img))
    # print(cv2.imwrite(debug.format("_smoothed"), sub_img))
    # print(cv2.imwrite(debug.format("_mask"), mask))
    # print(cv2.imwrite(debug.format("_shear"), sheared))
    # print(cv2.imwrite(debug.format("_rotate"), rotated))
  # log transform
  if logger is not None:
    logger.write(log + "\n")
  
  if output.shape[2] > 3:
    return output[:, :, :3], output[:, :, 3]
  else:
    return output

# def augment_one(data_dir = "/work/hpc/firedogs/data_/new_train/",
#                 img_name = "train_img_{0}.{1}",
#                 index = -1,
#                 output_dir = "/work/hpc/firedogs/potato/check/",
#                 log_dir = "/work/hpc/firedogs/potato/check/log.txt",
#                 rotate=(-40, 40), 
#                 shear_x=(-0.3, 0.3), 
#                 shear_y=(-0.2, 0.2), 
#                 noise=(0,), 
#                 debug= None, 
#                 keep_mask=False):
#       logger = open(log_dir, "w")
#       img_path = img_name.format(index, "jpg")
#       path = data_dir + img_path
#       img = cv2.imread(path)
#       if img is None:
#           flag = True
#           img_path = img_name.format(index, "png")
#           path = data_dir + img_paths
#           img = cv2.imread(path)
#       else:
#           print(path)
#       if img is None:
#           return 0
#       # print("Reading" + path)
#       output_path = output_dir + img_path
#       cv2.imwrite(output_path, augment_img(img, rotate=rotate, 
#                                             shear_x=shear_x, 
#                                             shear_y=shear_y, 
#                                             noise=noise, 
#                                             logger = logger,
#                                             debug= debug))
#       logger.close()


def augment_dir(data_dir = "/work/hpc/firedogs/data_/new_train/",
                img_name = "train_img_{0}.{1}",
                index = -1,
                output_dir = "/work/hpc/firedogs/potato/augmented_data/",
                log_dir = "/work/hpc/firedogs/potato/log.txt",
                rotate=(-40, 40), 
                shear_x=(-0.3, 0.3), 
                shear_y=(-0.2, 0.2), 
                noise=(0,)):
    logger = open(log_dir, "w")
    while True:
        flag = False
        index += 1
        img_path = img_name.format(index, "jpg")
        path = data_dir + img_path
        img = cv2.imread(path)
        if img is None:
            flag = True
            img_path = img_name.format(index, "png")
            path = data_dir + img_path
            img = cv2.imread(path)
        else:
            print(path)
        if img is None:
            break
        # print("Reading" + path)
        output_path = output_dir + img_path
        # print(output_path)
        cv2.imwrite(output_path, augment_img(img, rotate=rotate, 
                                                  shear_x=shear_x, 
                                                  shear_y=shear_y, 
                                                  noise=noise, 
                                                  logger = logger))
    logger.close()

