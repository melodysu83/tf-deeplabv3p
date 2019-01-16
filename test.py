import os
import numpy as np
import cv2
from model import *
from utils import *
import tensorflow as tf


def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        FLAGS.__delattr__(keys)
del_all_flags(tf.flags.FLAGS)
tf.reset_default_graph()


test_all=True
color_table = load_color_table('/home/fangbo-qin/sura-projects/BRL/sinus-segment/dataset/labels.json')
os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
run_config = tf.ConfigProto()
sess=tf.Session(config=run_config)
with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
    net = DeepLab(
          sess,
          input_width=320,
          input_height=240,
          batch_size=1,
          img_pattern="*.jpg",
          label_pattern="*.png",
          checkpoint_dir='./checkpoint_deeplabv3p_resnet50/DeepLab_16_240_320',
          pretrain_dir='',
          train_dataset='/home/fangbo-qin/sura-projects/BRL/sinus-segment/dataset/train',
          val_dataset='/home/fangbo-qin/sura-projects/BRL/sinus-segment/dataset/train',
          num_class=2,
          color_table=color_table,is_train=False)
    if not net.load(net.checkpoint_dir)[0]:
        raise Exception("Cannot find checkpoint!")
        
    
    
    if test_all:
    
        #test on train
        img_dir = "/home/fangbo-qin/sura-projects/BRL/sinus-segment/dataset/keyframes/"
        rst_dir = "/home/fangbo-qin/sura-projects/BRL/sinus-segment/dataset/train-rsts/"

        
        files=os.listdir(img_dir)
        for i,file in enumerate(files):
            if not file.endswith(".jpg"):
                continue
            
            
            img = cv2.imread(os.path.join(img_dir,file))
            
            t0 = time.time()
            idxmap,colormap = net.inference(img)  
            t = time.time()-t0
        
            print("{}, time={}sec".format(i,t))
            
            
            colormap=cv2.cvtColor(colormap,cv2.COLOR_RGB2BGR)
            
            cv2.imwrite(os.path.join(rst_dir,file),colormap)    
    
    else:
        imfile="/home/fangbo-qin/sura-projects/surgical-scene-segmentation/miccai_challenge_2018/train/images/seq_3_frame044.png"
        
        
        im=imread(imfile,resize_wh=(net.input_width,net.input_height),
                                         nearest_interpolate=True,grayscale=False)
                            
        idxmap,colormap = net.inference(im)  
        
        
        cv2.imshow("raw",im)
        cv2.imshow("toolnet",colormap)
        cv2.waitKey()