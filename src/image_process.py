# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016

"""SqueezeDet Demo.

In image detection mode, for a given image, detect objects and draw bounding
boxes around them. In video detection mode, perform real-time detection on the
video stream.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import time
import sys
import os
import glob

import numpy as np
import tensorflow as tf

from config import *
from train import _draw_box
from nets import *

from PIL import Image, ImageEnhance, ImageFilter

FLAGS = tf.app.flags.FLAGS
Sample = 'test_image'
ratio = 6		##low res compression ratio

tf.app.flags.DEFINE_string(
    'mode', 'image', """'image' or 'video'.""")
tf.app.flags.DEFINE_string(
    'checkpoint', './data/model_checkpoints/squeezeDet/model.ckpt-87000',
    """Path to the model parameter file.""")
tf.app.flags.DEFINE_string(
    'input_path', './data/'+ Sample +'.png',
    """Input image or video to be detected. Can process glob input such as """
    """./data/00000*.png.""")
tf.app.flags.DEFINE_string(
    'out_dir', './data/out/', """Directory to dump output image or video.""")


def image_demo():
  """Detect image."""
  for f in glob.iglob(FLAGS.input_path):
# 	img = Image.open(f).convert('L')
# 	img.save('./data/'+ Sample +'-gray.png','png')
# 	
# 	img = Image.open(f)
# 	width, height = img.size
# 	new_img = img.resize((width//ratio,height//ratio))
# # 	new_img = new_img.resize((width, height))
# 	new_img.save('./data/'+Sample+'-lowres.png','png')
# 	
# 	img = Image.open(f)
# 	contrast = ImageEnhance.Contrast(img)
# 	new_img = contrast.enhance(0.5)
# 	new_img.save('./data/'+Sample+'-lowcontrast.png','png')
# 	
# 	img = Image.open(f)
# 	contrast = ImageEnhance.Contrast(img)
# 	new_img = contrast.enhance(2)
# 	new_img.save('./data/'+Sample+'-highcontrast.png','png')
# 	
# 	img = Image.open(f)
# 	brightness = ImageEnhance.Brightness(img)
# 	new_img = brightness.enhance(0.5)
# 	new_img.save('./data/'+Sample+'-lowbright.png','png')
# 	
# 	img = Image.open(f)
# 	brightness = ImageEnhance.Brightness(img)
# 	new_img = brightness.enhance(2)
# 	new_img.save('./data/'+Sample+'-highbright.png','png')
# 	
# 	img = Image.open(f)
# 	color = ImageEnhance.Color(img)
# 	new_img = color.enhance(2)
# 	new_img.save('./data/'+Sample+'-highcolor.png','png')
# 
# 	img = Image.open(f)
# 	color = ImageEnhance.Color(img)
# 	new_img = color.enhance(0.5)
# 	new_img.save('./data/'+Sample+'-lowcolor.png','png')
# 	
	img = Image.open(f)
	new_img = img.filter(ImageFilter.GaussianBlur(2))
	new_img.save('./data/'+Sample+'-blur.png','png')
	
		
#   with tf.Graph().as_default():
# 	with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
# 	  saver.restore(sess, FLAGS.checkpoint)
# 
# 	  for f in glob.iglob(FLAGS.input_path):
# 		im = cv2.imread(f)
# 		im = im.astype(np.float32, copy=False)
# 		im = cv2.resize(im, (mc.IMAGE_WIDTH, mc.IMAGE_HEIGHT))
# 		input_image = im - mc.BGR_MEANS
                


def main(argv=None):
  if not tf.gfile.Exists(FLAGS.out_dir):
    tf.gfile.MakeDirs(FLAGS.out_dir)
  if FLAGS.mode == 'image':
    image_demo()
  else:
    video_demo()

if __name__ == '__main__':
    tf.app.run()
