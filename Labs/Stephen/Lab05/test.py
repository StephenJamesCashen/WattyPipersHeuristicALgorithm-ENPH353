import math
import numpy as np
import re
import os 
import subprocess
import string
from collections import Counter
from matplotlib import pyplot as plt
from PIL import Image
import cv2
"""## get data set labels"""

PATH = os.getcwd() + '/pictures'
file_names_raw = str(subprocess.check_output(['ls',PATH]))[2:]
file_names=file_names_raw.split('\\n')[:-1]

labels_raw = [file_name[6:10] for file_name in file_names]

def vectorize(label_raw):
  label =  [[0]*26, [0]*26,[0]*10,[0]*10]
  label[0][string.ascii_uppercase.index(label_raw[0])]=1
  label[1][string.ascii_uppercase.index(label_raw[1])]=1
  label[2][int(label_raw[2])]=1
  label[3][int(label_raw[3])]=1
  
  label = tuple(tuple(v) for v in label)
  return label

labels = np.array(tuple(vectorize(label_raw) for label_raw in labels_raw))


# Load the images
images = np.array([np.array(Image.open(PATH + '/' + img))
                    for img in file_names])



dataset = np.array([[images[i], labels[i]] for i in range(images.shape[0])])

data_amount = dataset.shape[0]
print("Loaded {:} images from folder:\n{}".format(data_amount, PATH))

blue=images[0, :, :, 0]
_, thresh = cv2.threshold(blue, 5, 255, cv2.THRESH_BINARY)

cv2.imshow('frame', thresh[75:-75, 50:150])

keyboard = cv2.waitKey(3000)
