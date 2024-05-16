
import numpy as np
import cv2
import random
import math
random.seed(None)

from src.data.components.aug.base import *

def sharp_mask(img, line_x_erosion, line_y_erosion, line_erode=True):
  """ Hard mask for morphology execution
  """
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

def augment_point(centroid, param):
  angle = param['angle']
  

def light_mask(img):
  """ Light mask for color processing
  """
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

def point_transform(points, M, homogeneous=False, transpose=False):  
  if homogeneous is False: 
    padded = np.pad(points, ((0, 0), (0, 1)), mode='constant', constant_values=(1,))
  else:
    padded = points
  
  if transpose is True:
    padded[:, :2] = np.roll(padded[:, :2], 1, axis=1)
  transformed = np.vstack([M @ p for p in padded])
  
  if transpose is True:
    transformed[:, :2] = np.roll(transformed[:, :2], -1, axis=1)
  
  if homogeneous is True:
    return transformed
  else:
    return transformed[:, :2]

# def warp_image(img, variance):
  
  
def warp_transform(img, 
                   mask,
                   scale, 
                   angle, 
                   shear_x, 
                   shear_y, 
                   translate_x, translate_y, 
                   pad_x, pad_y, 
                   bg_color, 
                   export=False, 
                   alpha=False,
                   border_replicate=False):
  """ Shape transform wrapper by fore-matmul all transformation 
  """

  # shear transformation
  shear_kernel = np.array([ [1, shear_x, 0],
                            [shear_y, 1, 0],
                            [0, 0, 1]])
  h, w = img.shape[:2]
  
  # skew transformation
  skew = cv2.getRotationMatrix2D((int((w + h * shear_x) / 2), int((h + w * shear_y) / 2)), angle=-angle, scale=1.)
  skew_pad = np.pad(skew, ((0, 1), (0, 0)), mode='constant', constant_values=(0,))
  skew_pad[2, 2] = 1
  
  transform_matrix = skew_pad @ shear_kernel
  
  # padding
  corner = np.zeros((4, 2))
  corner[1:3, 1] += h
  corner[2:4, 0] += w

  # transform image corner for output size configuration
  transformed_corner = point_transform(corner, transform_matrix, homogeneous=False)
  transform_matrix[:, 2] -= [min(transformed_corner[:, 0]), min(transformed_corner[:, 1]), 0]
  transform_matrix[:, 2] += [translate_y, translate_x, 0]
  w_new, h_new= (np.max(transformed_corner, axis=0) - np.min(transformed_corner, axis=0)).astype(int) + [translate_y + pad_y + 1, translate_x + pad_x + 1]
  
  masked_img = np.concatenate([img, mask[:, :, None].astype(np.uint8)], axis=2)
  masked_img = cv2.resize(masked_img, (0, 0),fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

  if border_replicate is False:
    transformed = cv2.warpPerspective(masked_img, 
                                      transform_matrix, 
                                      dsize=(w_new, h_new), 
                                      borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=bg_color, 
                                      flags=cv2.INTER_LINEAR)
  else:
    transformed_img = cv2.warpPerspective(masked_img[:, :, :3], 
                                          transform_matrix, 
                                          dsize=(w_new, h_new), 
                                          borderMode=cv2.BORDER_REPLICATE,
                                          flags=cv2.INTER_LINEAR)
    transformed_mask = cv2.warpPerspective(masked_img[:, :, 3], 
                                          transform_matrix, 
                                          dsize=(w_new, h_new), 
                                          borderMode=cv2.BORDER_CONSTANT,
                                          borderValue=(0,), 
                                          flags=cv2.INTER_LINEAR)
    transformed = np.concatenate([transformed_img, transformed_mask[:, :, None].astype(np.uint8)], axis=2)
  
  if alpha is True:
    alpha_mask = cv2.warpPerspective(np.ones((img.shape[0], img.shape[1]), dtype=np.float32), 
                                transform_matrix, 
                                dsize=(w_new, h_new), 
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(0,), 
                                flags=cv2.INTER_LINEAR)
  else:
    alpha_mask = None
  
  return transformed[:, :, :img.shape[2]], transformed[:, :, img.shape[2]], transform_matrix, alpha_mask
  
  
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
                keep_mask = False, 
                export= False, 
                borderMode='constant'):
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

  mask = np.max((img < thresh).astype(int), axis = 2).astype(np.uint8)

  # trich xuat bg
  bg_mask = img < thresh
  bg_color = extract_median_color_axis(img.copy(), bg_mask, axis = (0, 1)).tolist()
  # bg_color = [0, 0, 0]
  bg_color.append(0)

  # # kéo 
  # sheared, sheared_mask = shear_img(img, mask, shear_x_level, shear_y_level, bg_color)
  # # xoay
  # rotated, rotated_mask = rotate_img(sheared, skew_angle, sheared_mask, bg_color)
  # # cắt
  
  transformed, transformed_mask, transform_matrix = warp_transform(img, mask, skew_angle, shear_x_level, shear_y_level, bg_color, export=export)
  
  if keep_mask is True:
    merged = np.concatenate([transformed, transformed_mask[:, :, None].astype(np.uint8)], axis=2)
    output, transform_matrix = crop_img(merged, transformed_mask, 0, 0, 0, 0, bg_color, smoothing = False, transform=transform_matrix)
  else:
    output, transform_matrix = crop_img(transformed, transformed_mask, 0, 0, 0, 0, bg_color[:3], smoothing = False, transform=transform_matrix)
    
  # log anh ra ngoai
  if debug is not None:
    print("Debug_folder: " + debug)
    mask_img = mask * 255
    print(mask_img.shape, mask_img.dtype)
    print(cv2.imwrite(debug.format("_original"), img))
    print(cv2.imwrite(debug.format("_smoothed"), sub_img))
    print(cv2.imwrite(debug.format("_mask"), mask))
    # print(cv2.imwrite(debug.format("_shear"), sheared))
    print(cv2.imwrite(debug.format("_transform"), transformed))
    
  # log transform
  if logger is not None:
    logger.write(log + "\n")
  
  return  output[:, :, :img.shape[2]], (lambda x: output[:, :, img.shape[2]] if x is True else None)(output.shape[2] > img.shape[2]), transform_matrix

def augment_one(data_dir = "/work/hpc/firedogs/data_/new_train/",
                img_name = "train_img_{0}.{1}",
                index = -1,
                output_dir = "/work/hpc/firedogs/potato/check/",
                log_dir = "/work/hpc/firedogs/potato/check/log.txt",
                rotate=(-40, 40), 
                shear_x=(-0.3, 0.3), 
                shear_y=(-0.2, 0.2), 
                noise=(0,), 
                debug= None, 
                keep_mask=False):
      logger = open(log_dir, "w")
      img_path = img_name.format(index, "jpg")
      path = data_dir + img_path
      img = cv2.imread(path)
      if img is None:
          flag = True
          img_path = img_name.format(index, "png")
          path = data_dir + img_paths
          img = cv2.imread(path)
      else:
          print(path)
      if img is None:
          return 0
      # print("Reading" + path)
      # output = augment_img(img,  
      #                     rotate=rotate, 
      #                     shear_x=shear_x, 
      #                     shear_y=shear_y, 
      #                     noise=noise, 
      #                     logger = logger)
      # print(output)
        
        
      output_path = output_dir + img_path
      cv2.imwrite(output_path, augment_img(img, rotate=rotate, 
                                            shear_x=shear_x, 
                                            shear_y=shear_y, 
                                            noise=noise, 
                                            logger = logger,
                                            debug= debug)[0])
      logger.close()


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
        print("Reading" + path)
        output_path = output_dir + img_path
        print(output_path)
        output = augment_img(img,  
                          rotate=rotate, 
                          shear_x=shear_x, 
                          shear_y=shear_y, 
                          noise=noise, 
                          logger = logger)
        print(output)
        # cv2.imwrite(output_path, augment_img(img, rotate=rotate, 
        #                                           shear_x=shear_x, 
        #                                           shear_y=shear_y, 
        #                                           noise=noise, 
        #                                           logger = logger))
    logger.close()


if __name__ == "__main__":
    augment_one(index = 2315, rotate=(-30, 30), shear_x=(-0.8, 0.8), shear_y=(0.8,), debug = "/work/hpc/firedogs/potato/check/img{}.jpg")
    # augment_dir()
# py /work/hpc/firedogs/potato/augment.py
# /work/hpc/firedogs/data_old/new_train/train_img_94093.png