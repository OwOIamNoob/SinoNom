import numpy as np
import cv2
import random
import math
import subprocess
import os

from base import *
from wrapper_v2 import *
import deaug
# texture = load_texture("/work/hpc/firedogs/potato/asset/texture.png", intensity=2)
#################### CHARACTER AND PUNCTUATION RECOGNITION ############################
def find_centroid(mask, bboxes, wh=False):
  centroid = []
  for i in range(len(bboxes)):
    x, y, w, h = bboxes[i]
    center = np.median(np.where(mask[y:y+h, x:x+w] > 0), axis=1) + [y, x]
    # #print(center)
    centroid.append(center)
  # plt.imshow(img)
  if wh is True:
    return 
  return centroid

def regression(points2d):
  x = np.ones((points2d.shape[0], 2))
  x[:, 0] = points2d[:, 1]
  param = np.linalg.lstsq(x, points2d[:, 0], rcond=None)[0]
  return param

def find_text_and_punc(img, mask = None, label="", ambiguous=True):
  if mask is None:
    mask = deaug.sharp_mask(img, 0.8, 0.8)

  print("Mask shape:",mask.shape, mask.dtype)
  cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = np.array(cnts[len(cnts) % 2])
  #print(cnts.shape, " Contours", cnts[0].shape)
  
  punc = 0
  for char in label:
    if char == '^' or char == '~' or char == '`' or char == '\'' or char == '?' or char == '.' or char == '_':
      punc += 1
  num_char = len(label) - punc
  
  ratio = np.sum(mask) / (num_char * 1.2 + punc * 0.5)
  #print("Threshold: ", ratio)

  boxes = np.array([cv2.boundingRect(cnt) for cnt in cnts])
  #print(boxes.shape[0])

  is_char = np.full((len(cnts),), False, dtype=bool)
  for i in range(boxes.shape[0]):
    x, y, w, h = boxes[i]
    convex = cv2.convexHull(cnts[i])
    # #print(convex)
    area = cv2.contourArea(convex)

    if area > 0.4 * h * w and ((h >= img.shape[0] * 0.1) or (w >= img.shape[1] * 0.1)):
      if np.sum(mask[y: y + h, x: x + w]) > ratio :
        is_char[i] = True
        #print("Its a char")
      else:
        min_rect = cv2.minAreaRect(cnts[i])
        if min_rect[2] < 10 and min_rect[1][0] * 2 < min_rect[1][1]:
          is_char[i] = True
          #print("Straight-up char")
        else:
          is_char[i]= False
          #print("Not promising")
    else:
      #print(area)
      is_char[i] = False
      #print("Not filling")
  #print(is_char)
  #print(boxes[is_char == True].shape)

  centroid = None
  centroid = np.array(find_centroid(mask, boxes))

  if np.sum(is_char) >= 1:
    if np.sum(is_char) > 1:
      param = regression(centroid[is_char == True])
      
    else:
      param = [0, centroid[is_char==True][0, 0]]

    print(param)
    #print(param)
    max_dist = np.max(np.abs(np.sum(centroid[is_char == True] * [-1, param[0]], axis=1) + param[1]) / np.sqrt(param[0] * param[0] + 1))
    punc_dist = np.abs(np.sum(centroid[is_char == False] * [-1, param[0]], axis=1) + param[1]) / np.sqrt(param[0] * param[0] + 1)
    punc_loc = np.where(is_char == False)[0]
    #print(punc_loc)
    #print(punc_dist, " Maximum:", max_dist)
    for i in range(len(punc_loc)):
        if punc_dist[i] < max(max(img.shape[0], img.shape[1]) / 10, max_dist):
          is_char[punc_loc[i]] = True

  #print("There are {0} text and {1} punctuation".format(np.sum(is_char.astype(int)), len(label) - np.sum(is_char.astype(int))))
  return cnts, boxes, is_char, centroid

def line_vector(boxes,
                is_char,
                centroid,
                img_size,
                transform = None,
                y_noise = 0,
                skew_noise = 0,
                intent = False,
                align = 0,
                ignore_skew = False,
                axis = 1):
  a, b = 0, 0
  gather = False
  if axis == 0:
    ignore_skew = True
  if np.sum((is_char == False).astype(int)) == 0:
    intent = False
    
  if np.sum((is_char == True).astype(int)) < 2:
    if len(is_char) == 1:
      ignore_skew = True
    else:
      is_char[:] = True
    
  # print(intent, ignore_skew)
  
  if intent is True:
    punc_loc = np.array(np.where(is_char == False))
    index = np.random.choice(punc_loc.flatten())
    if axis == 0:
      a = np.random.normal(0, skew_noise)
      b = centroid[index, 1] - centroid[index, 0] * a
    else:
      a = np.random.normal(0, skew_noise)
      b = centroid[index, 0] - centroid[index, 1] * a
    return a, b

  else:
    baseline = centroid[is_char == True]
    baseline[:, 0] += np.mean(np.min(boxes[is_char == True, 2:4], axis = 1)) * align
    if transform is not None:
      baseline = point_transform(baseline, transform, transpose=True)
    
    a, b = regression(baseline)
    # bound = np.mean(np.min(boxes[is_char == True, 2:4], axis = 1))
    # b += bound * align
    b += np.random.normal(0, y_noise)
    
    if ignore_skew is True:
      x = np.random.randint(0, img_size[1])
      y = a * x + b
      if axis == 0:
        A = np.random.normal(0, skew_noise)
        B = x - y * a
      else:
        A = np.random.normal(0, skew_noise)
        B = y - x * a
      return A, B
    else:
      a += np.random.normal(0, skew_noise)
      return a, b

def draw_line(img, pattern, spacing, A, b, axis=1, noise_level=0):
  # #print(img.shape, pattern.shape)
  # padding = int(spacing + max(pattern.shape))
  if axis == 1:
    x = np.arange(0, img.shape[1] - pattern.shape[1], max(spacing, pattern.shape[1]), dtype=int)  
    y = (x * A + b).astype(int)
    # #print(y) -t 
  else:
    y = np.arange(0, img.shape[0] - pattern.shape[0], max(spacing, pattern.shape[0]), dtype=int)
    x = (y * A + b).astype(int)
    # #print(x)
  
  for i in range(x.shape[0]):
    if x[i] < - pattern.shape[1] or y[i] < - pattern.shape[0] or x[i] > img.shape[1] or y[i] > img.shape[0]:
      continue
    shape = img[y[i]:y[i] + pattern.shape[0], x[i]:x[i] + pattern.shape[1]].shape
    img[y[i]:y[i] + shape[0], x[i]:x[i] + shape[1]] =  ( img[y[i]:y[i] + shape[0], x[i]:x[i] + shape[1]].astype(float) * pattern[:shape[0], :shape[1]] )
    
  return img

##### WRAPPER 
def line_and_noise(img, label, param=None, mask=None, transform=None, bg_color = 1, line_width = 2, y_noise=0,skew_noise=0, intent=False, align=0, ignore_skew=False, axis=1, spacing=0, noise_level=0):
    if param is not None:
      cnts = param['cnts']
      boxes = param['boxes']
      is_char = param['is_char']
      centroid = param['centroid']
    else:
      cnts, boxes, is_char, centroid = find_text_and_punc(img, mask, label)

    a, b = line_vector(cnts, boxes, is_char, centroid, img.shape, y_noise, skew_noise, intent=intent, align=align, ignore_skew=ignore_skew, axis=axis)
    customized_pattern = cv2.resize(texture, (line_width, line_width), interpolation=cv2.INTER_LINEAR)
    output = draw_line(img, customized_pattern, spacing, a, b, axis=axis, noise_level=noise_level)
    return output
 
#### TEST
if __name__ =="abc":
    augmenter = Augmenter("./potato/asset/translate.txt", "./potato/asset/texture.png")
    img_dir = "/work/hpc/firedogs/data_/new_train/"
    img_label = "/work/hpc/firedogs/data_/train_gt.txt"
    output_dir = "/work/hpc/firedogs/potato/asset/line/"
    param_file = "/work/hpc/firedogs/potato/asset/line/{}.npz"
    with open(img_label, "r") as file:
        for line in file:
            parts = line.strip().split()
            img = cv2.imread(img_dir + parts[0])
            filename = parts[0].split(".")[0]
            print(filename)
            param = np.load(param_file.format(filename), allow_pickle=True)
            translated = augmenter.translate(parts[1])
            cv2.imwrite("/work/hpc/firedogs/potato/output/testing_{}.jpg".format(filename), line_and_noise(img, label =translated,
                                                                                                  param=param, 
                                                                                                  mask=None,
                                                                                                  bg_color=None,
                                                                                                  line_width=2,
                                                                                                  y_noise=4,
                                                                                                  intent=True,
                                                                                                  align=-1,
                                                                                                  ignore_skew=False,
                                                                                                  axis=1,
                                                                                                  spacing=2,
                                                                                                  noise_level=0.2))

if __name__ =="abc":
  augmenter = Augmenter("./potato/asset/translate.txt", "./potato/asset/texture.png")
  checkpoint_dir =  "/work/hpc/firedogs/potato/asset/line/"
  id = 353
  filename = "train_img_{}.npz".format(id)
  img_label = "/work/hpc/firedogs/data_/train_gt.txt"
  
  
  data = np.load(checkpoint_dir + filename, allow_pickle=True)
  img = cv2.imread("/work/hpc/firedogs/data_/new_train/train_img_{0}.{1}".format(id, "jpg"))
  print(data['boxes'].shape)
  print(data['centroid'])
  # cv2.imwrite("/work/hpc/firedogs/potato/output/testing_saved.jpg", line_and_noise(img,label ="BỆN",param=data, mask=None,
                                                                                                    # bg_color=None,
                                                                                                    # line_width=2,
                                                                                                    # y_noise=4,
                                                                                                    # intent=True,
                                                                                                    # align=-1,
                                                                                                    # ignore_skew=False,
                                                                                                    # axis=1,
                                                                                                    # spacing=2,
                                                                                                    # noise_level=0.2))

if __name__ == "__main__":
    augmenter = Augmenter("/work/hpc/firedogs/potato/public_test_optim/translate.txt", "./potato/asset/texture.png", "./potato/asset/line/", "")
    img_dir = "/work/hpc/firedogs/data_/new_public_test/"
    img_label = "/work/hpc/firedogs/BKAI_private/data/vietocr_private_data/public_gt_22939.txt"
    output = "/work/hpc/firedogs/potato/public_test_optim/line/{}.npz"
    log_file = open("/work/hpc/firedogs/potato/public_test_optim/line/log.txt", "w")
    fnames = np.loadtxt(img_label, dtype=str)
    # fnames = np.unique(fnames)
    
    for fname in fnames:
        # parts = line.strip().split()
        img = cv2.imread(img_dir + fname[0])
        print(fname)
        filename = fname[0].split(".")[0]
        translated = Augmenter.translate(fname[1])
        _, boxes, is_char, centroid = find_text_and_punc(img, None, translated)
        a = np.sum((is_char == True).astype(int))
        log_file.write(filename + " has {} char and {} punc out of label {}".format(a, len(translated) - a, translated))
        print(output.format(filename))
        np.savez(output.format(filename), boxes=boxes, is_char=is_char, centroid=centroid)
    log_file.close()