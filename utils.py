"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import json
import random
import pprint
import scipy.misc
import numpy as np
from time import gmtime, strftime
from six.moves import xrange
import cv2

import tensorflow as tf
import tensorflow.contrib.slim as slim

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])


def load_color_table(json_file):
    # load color table
    f= open(json_file, "r", encoding='utf-8')
    colors = json.loads(f.read())
    class_num=len(colors)
    R,G,B=[[],[],[]]
    for c in colors:
        R.append(c['color'][0])
        G.append(c['color'][1])
        B.append(c['color'][2])
    return [R,G,B]        
    
def idxmap2colormap(im_idx,color_table):
    R,G,B = color_table
    class_num = len(R)
    imR = np.zeros_like(im_idx,np.uint8)
    imG = np.zeros_like(im_idx,np.uint8)
    imB = np.zeros_like(im_idx,np.uint8)
    for i in range(class_num):
        imR[im_idx==i]=R[i]
        imG[im_idx==i]=G[i]
        imB[im_idx==i]=B[i]
    imcolor = np.dstack((imR,imG,imB))
    return imcolor

def show_all_variables():
  model_vars = tf.trainable_variables()
  slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def save_images(images, size, image_path):
  return imsave(images, size, image_path)

def imread(path,resize_wh=None, nearest_interpolate=False, grayscale = False):
  image = cv2.imread(path)
  if grayscale and image.shape[2]>0:
      image = image[:,:,0]
  if resize_wh is not None:
      if nearest_interpolate:
          image = cv2.resize(image,resize_wh,interpolation=cv2.INTER_NEAREST)
      else:
          image = cv2.resize(image,resize_wh)
  return image


def imsave(images, size, path):
  image = np.squeeze(merge(images, size))
  return scipy.misc.imsave(path, image)

def center_crop(x, crop_h, crop_w,
                resize_h=64, resize_w=64):
  if crop_w is None:
    crop_w = crop_h
  h, w = x.shape[:2]
  j = int(round((h - crop_h)/2.))
  i = int(round((w - crop_w)/2.))
  return scipy.misc.imresize(
      x[j:j+crop_h, i:i+crop_w], [resize_h, resize_w])




