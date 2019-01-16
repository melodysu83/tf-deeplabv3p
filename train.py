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


flags = tf.app.flags
flags.DEFINE_integer("epoch",50, "Epoch to train [25]")
flags.DEFINE_integer("batch_size", 16, "The size of batch images [64]")
flags.DEFINE_integer("input_height", 240, "The size of image to use (will be center cropped). [108]")
flags.DEFINE_integer("input_width",320, "The size of image to use (will be center cropped). If None, same value as input_height [None]")
flags.DEFINE_string("train_dataset", "/home/fangbo-qin/sura-projects/BRL/sinus-segment/dataset/train", "train dataset direction")
flags.DEFINE_string("val_dataset", "/home/fangbo-qin/sura-projects/BRL/sinus-segment/dataset/train", "train dataset direction")
flags.DEFINE_string("img_pattern", "*.jpg", "Glob pattern of filename of input images [*]")
flags.DEFINE_string("label_pattern", "*.png", "Glob pattern of filename of input labels [*]")
flags.DEFINE_string("checkpoint_dir", "./checkpoint_deeplabv3p_resnet50_mse2/", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("pretrain_dir", "./pretrain", "")
FLAGS = flags.FLAGS

def main(_):
  pp.pprint(flags.FLAGS.__flags)

  if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)

  color_table = load_color_table('/home/fangbo-qin/sura-projects/BRL/sinus-segment/dataset/labels.json')
  #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
  run_config = tf.ConfigProto()
  run_config.gpu_options.allow_growth=True
  tf.reset_default_graph()
  with tf.Session(config=run_config) as sess:

    net = DeepLab(
          sess,
          input_width=FLAGS.input_width,
          input_height=FLAGS.input_height,
          batch_size=FLAGS.batch_size,
          img_pattern=FLAGS.img_pattern,
          label_pattern=FLAGS.label_pattern,
          checkpoint_dir=FLAGS.checkpoint_dir,
          pretrain_dir=FLAGS.pretrain_dir,
          train_dataset=FLAGS.train_dataset,
          val_dataset=FLAGS.val_dataset,
          num_class=2,
          color_table=color_table,is_train=True)

    net.train(FLAGS)

      
      

    
if __name__ == '__main__':
  tf.app.run()
