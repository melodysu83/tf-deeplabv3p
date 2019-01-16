from __future__ import division
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
import random
import cv2
from resnetv2 import *
slim = tf.contrib.slim


try:
    from .ops import *
    from .utils import *
except:    
    from ops import *
    from utils import *


class DeepLab(object):
  def __init__(self,sess,
          input_width,
          input_height,
          batch_size,
          img_pattern,
          label_pattern,
          checkpoint_dir,
          pretrain_dir,
          train_dataset,
          val_dataset,
          num_class,
          color_table,
          is_train=False):

    self.sess = sess
    self.is_train=is_train


    self.batch_size = batch_size
    self.num_class = num_class
    self.input_height = int(input_height)
    self.input_width = int(input_width)
    self.chn = 3

    self.learning_rate=0.001
    self.beta1=0.9

    self.model_name = "resnet_v2_50"

    self.train_dataset = train_dataset
    self.val_dataset = val_dataset
    self.img_pattern = img_pattern
    self.label_pattern = label_pattern
    self.checkpoint_dir = checkpoint_dir
    self.pretrain_dir = pretrain_dir
    self.color_table = color_table

    self.data = glob(os.path.join(self.train_dataset,"images", self.img_pattern))
    self.label = glob(os.path.join(self.train_dataset,"labels", self.label_pattern))
    self.val_data = glob(os.path.join(self.val_dataset,"images", self.img_pattern))
    self.val_label = glob(os.path.join(self.val_dataset,"labels", self.label_pattern))
    
    self.data.sort()
    self.label.sort()
    self.val_data.sort()
    self.val_label.sort()

    self.build_model()
    self.build_augmentation()
    
    
  def build_augmentation(self):
    image_dims = [self.input_height, self.input_width, self.chn]
    label_dims = [self.input_height, self.input_width,1]
        
    # augmentation modual
    self.im_raw = tf.placeholder(tf.float32,  image_dims, name='im_raw')
    self.label_raw = tf.placeholder(tf.int32, label_dims, name='label_raw')
    seed =23   
    def augment(image,color=False):
      r = image
      if color:
        r/=255.
        r = tf.image.random_hue(r,max_delta=0.1, seed=seed)
        r = tf.image.random_brightness(r,max_delta=0.3, seed=seed)
        r = tf.image.random_saturation(r,0.2,1.2, seed=seed)
        r = tf.image.random_contrast(r,0.3,1.3, seed=seed)
        r = tf.minimum(r, 1.0)
        r = tf.maximum(r, 0.0)
        r*=255.
      r = tf.image.random_flip_left_right(r, seed=seed)
      r = tf.image.random_flip_up_down(r,seed=seed)
      r = tf.contrib.image.rotate(r,tf.random_uniform((), minval=-np.pi/180*90,maxval=np.pi/180*90,seed=seed),interpolation='NEAREST')
      
      return r
    
    self.im_aug,self.label_aug = [augment(self.im_raw,color=True),augment(self.label_raw)]
    

  def build_model(self):

    ##############################################################
    # inputs
    image_dims = [self.input_height, self.input_width, self.chn]
    label_dims = [self.input_height, self.input_width]

    self.inputs = tf.placeholder(tf.float32, [self.batch_size] + image_dims, name='image')
    self.targets = tf.placeholder(tf.int32, [self.batch_size] + label_dims, name='label')
    self.keep_prob = tf.placeholder(tf.float32)
    
    ################################################################
    # layers
    layers = []


    h = self.inputs-127.5 #512
    
    
    end_points = {}
    with slim.arg_scope([slim.batch_norm, slim.dropout],is_training=self.is_train):
        _, end_points = resnet_v2_50(h,
             num_classes=0,
             is_training=self.is_train,
             global_pool=False,
             output_stride=32,
             spatial_squeeze=False,
             reuse=None,
             scope=self.model_name)
    h = end_points['resnet_v2_50/block3/unit_4/bottleneck_v2/conv3']         
    skip = end_points["resnet_v2_50/block1/unit_3/bottleneck_v2/conv1"]

        
        
    h = tf.nn.dropout(h,self.keep_prob)
    
    
    # atrous spatial pyramid pooling
    h = atrous_spatial_pyramid_pooling(h, output_stride=16, depth=256,is_train=self.is_train)
    # upsample*4
    h = tf.image.resize_bilinear(h, tf.shape(skip)[1:3])
    
    # skip connect low level features
    skip = conv2d(skip,32,ksize=1,stride=1,name="conv_skip") 
    skip = tf.nn.relu(batchnorm(skip,self.is_train,'bn_skip'))
    
    # concate and segment
    h = tf.concat([h,skip],axis=3)
    
    h = tf.nn.dropout(h,self.keep_prob)
    
    h = separable_conv2d(h,256,ksize=3,name="conv_out1")
    h = tf.nn.relu(batchnorm(h,self.is_train,'bn_out1'))
    
    h = separable_conv2d(h,256,ksize=3,name="conv_out2")
    h = tf.nn.relu(batchnorm(h,self.is_train,'bn_out2')) 
    
    h = conv2d(h,self.num_class,ksize=3,stride=1,name="conv_out3")
    # upsample
    h = tf.image.resize_bilinear(h, [self.input_height, self.input_width])
    ###########################################################################
    output_logits = h
    self.output_softmax = tf.nn.softmax(output_logits)
    self.output = tf.cast(tf.argmax(self.output_softmax,axis=3),tf.uint8,name='outputs')
    
    
    ###########################################################################
    #loss
    K=self.num_class
    label_map = tf.one_hot(tf.cast(self.targets,tf.int32),K)
    
     
    flat_label = tf.reshape(label_map,[-1,K])
    flat_out = tf.reshape(self.output_softmax,[-1,K])
    self.seg_loss = tf.reduce_mean(tf.square(flat_label-flat_out))
        
        
    self.loss_sum = scalar_summary("loss", self.seg_loss)
    self.val_loss_sum = scalar_summary("val_loss", self.seg_loss)
    # saver
    g_vars = tf.global_variables()
    bn_moving_vars = [g for g in g_vars if 'moving_' in g.name]
    self.tvars=tf.trainable_variables()
    self.load_vars = [var for var in self.tvars if self.pretrain_dir is '' or (self.model_name in var.name and "biases" not in var.name)]
    self.saver = tf.train.Saver(self.tvars+bn_moving_vars,max_to_keep=50)
    if self.pretrain_dir is '':
        self.load_vars+=bn_moving_vars
    self.loader = tf.train.Saver(self.load_vars)
    
    
  def inference(self, img):
    shape0 = (img.shape[1],img.shape[0])
    img=cv2.resize(img,(self.input_width,self.input_height))
    
    
    inputs = np.array([img]).astype(np.float32)

    out_softmax = self.sess.run(self.output_softmax,feed_dict={self.inputs:inputs,self.keep_prob:1.0})
    out = out_softmax[0,:,:,1]
    out[out>=0.1]=1
    out[out<0.1]=0
    
    
    
    rst=idxmap2colormap(out,self.color_table)
    
    idxmap = cv2.resize(out,shape0,interpolation=cv2.INTER_NEAREST)
    colormap = cv2.resize(rst,shape0,interpolation=cv2.INTER_NEAREST)
    
    return idxmap,colormap
      
      
      
  def train(self,config):

    batch_num = len(self.data) // self.batch_size
    
    # learning rate
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = self.learning_rate
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           batch_num*config.epoch/4, 0.5, staircase=True)
      
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # for updating moving average of batchnorm
    with tf.control_dependencies(update_ops):
      optim = tf.train.AdamOptimizer(learning_rate, beta1=self.beta1) \
              .minimize(self.seg_loss, var_list=self.tvars,global_step=global_step)
        
    
    try:
      tf.global_variables_initializer().run()
    except:
      tf.initialize_all_variables().run()

    self.train_sum = merge_summary([self.loss_sum])
    self.writer = SummaryWriter(os.path.join(self.checkpoint_dir,"logs"), self.sess.graph)
  
    counter = 1
    start_time = time.time()
    if os.path.exists(self.pretrain_dir):
        could_load, checkpoint_counter = self.load_pretrain(self.pretrain_dir)
    else:
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
    if could_load:
      counter = checkpoint_counter
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")


    idxs = np.arange(len(self.data))
    idxv = np.arange(len(self.val_data))
    for epoch in xrange(config.epoch):
      
      random.shuffle(idxs)
      random.shuffle(idxv)
      
      for idx in xrange(0, batch_num):
        file_idxs = idxs[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_images = [imread(self.data[i],resize_wh=(self.input_width,self.input_height),
                                     nearest_interpolate=True,grayscale=False) for i in file_idxs]
        batch_labels = [imread(self.label[i],resize_wh=(self.input_width,self.input_height),
                                     nearest_interpolate=True,grayscale=True) for i in file_idxs]
                                     
        # augmentaion
        batch_images = [self.sess.run(self.im_aug,feed_dict={self.im_raw:im}) for im in batch_images]
        batch_labels = [self.sess.run(self.label_aug,feed_dict={self.label_raw:np.reshape(lb,[lb.shape[0],lb.shape[1],1])})[:,:,0] for lb in batch_labels]
        
        batch_images = np.array(batch_images).astype(np.float32)
        batch_labels = np.array(batch_labels).astype(np.int32)
        # Update gradient
        _,train_loss, summary_str,cur_lr = self.sess.run([optim,self.seg_loss, self.loss_sum,learning_rate],
                                                  feed_dict={ self.inputs: batch_images, self.targets: batch_labels,self.keep_prob:0.8})
        self.writer.add_summary(summary_str, counter)

        counter += 1
        print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.2f, loss: %.8f, lr: %.8f" \
          % (epoch, config.epoch, idx, batch_num, time.time() - start_time, train_loss, cur_lr))

        if counter% (batch_num//10) == 0:
          file_idx0 = np.random.randint(len(self.val_data)-self.batch_size)
          file_idxs = idxv[file_idx0:self.batch_size+file_idx0]
          val_batch_images = [imread(self.val_data[i],resize_wh=(self.input_width,self.input_height),
                                     nearest_interpolate=True,grayscale=False) for i in file_idxs]
          val_batch_labels = [imread(self.val_label[i],resize_wh=(self.input_width,self.input_height),
                                     nearest_interpolate=True,grayscale=True) for i in file_idxs]
                        
          val_batch_images = np.array(val_batch_images).astype(np.float32)
          val_batch_labels = np.array(val_batch_labels).astype(np.int32)
          out, train_loss, summary_str = self.sess.run([self.output,self.seg_loss, self.val_loss_sum],
                                                  feed_dict={ self.inputs: val_batch_images, self.targets: val_batch_labels,self.keep_prob:1.0})
          self.writer.add_summary(summary_str, counter)
          
          disp_idx=(counter//(batch_num//10))%self.batch_size
          output=idxmap2colormap(out[disp_idx,:,:],self.color_table)
          label = idxmap2colormap(val_batch_labels[disp_idx,:,:],self.color_table)
          input = val_batch_images[disp_idx,:,:,:]
          rst=np.hstack((input,label,output))
          filename = "%08d.png" % (counter)
          
          label=cv2.cvtColor(label,cv2.COLOR_RGB2BGR)
          output=cv2.cvtColor(output,cv2.COLOR_RGB2BGR)
          cv2.imwrite(os.path.join(self.checkpoint_dir,filename),rst)
          
          
        if np.mod(counter, (batch_num//2)) == 0:
          self.save(self.checkpoint_dir, counter)    
    self.save(self.checkpoint_dir, counter)    

          
  @property
  def model_dir(self):
    return "{}_{}_{}_{}".format(
        'DeepLab', self.batch_size,
        self.input_height, self.input_width)
      
  def save(self, checkpoint_dir, step):
    model_name = "DeepLab.model"
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
            os.path.join(checkpoint_dir, model_name),
            global_step=step)

  def load(self, checkpoint_dir):
    import re
    print(" [*] Reading checkpoints...")
    try:
      self.sess.run(tf.global_variables_initializer())
    except:
      self.sess.run(tf.initialize_all_variables().run())
    
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.loader.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
      counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
      print(" [*] Success to read {}".format(ckpt_name))
      return True, counter
    else:
      print(" [*] Failed to find a checkpoint")
      return False, 0
    
  def load_pretrain(self, pretrain_file):
    import re
    print(" [*] Reading checkpoints...")
    try:
      self.sess.run(tf.global_variables_initializer())
    except:
      self.sess.run(tf.initialize_all_variables().run())
    
    
    self.loader.restore(self.sess,os.path.join(self.pretrain_dir,self.model_name+".ckpt"))
#    tf.train.init_from_checkpoint(self.pretrain_file,{v.name.split(':')[0]:v for v in self.load_vars})
    counter = 0
    return True,counter
      
