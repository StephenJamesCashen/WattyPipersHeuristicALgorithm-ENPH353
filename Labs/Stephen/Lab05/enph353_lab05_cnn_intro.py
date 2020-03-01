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

PATH = 'Lab05/pictures'
file_names_raw = str(subprocess.check_output(['ls',PATH]))[2:]
file_names=file_names_raw.split('\\n')[:-1]

labels_raw = np.array([file_name[6:10] for file_name in file_names])

y_labels = []
for l in labels_raw:
  y_labels.extend(l)

def num_from_label(label):
  if ord(label) <= 57:
    return ord(label) - 48
  return ord(label) - 65 + 10

y_labels = np.array([num_from_label(l) for l in y_labels])
def vectorize(label_raw):
  label =  [[0]*36, [0]*36,[0]*36, [0]*36]
  for i in range(0,2):
    label[i][ord(label_raw[i])-55]=1

  for i in range(2,4):
    label[i][ord(label_raw[i])-48]=1

  return label

labels = np.array([vectorize(label_raw) for label_raw in labels_raw])

# Load the images (BGR format)
images_raw = np.array([np.array(Image.open(PATH + '/' + img))
                    for img in file_names])
                 

print("Loaded {:} images from folder:\n{}".format(images_raw.shape[0], PATH))

def proccess_and_split(img):
  
  blue = np.split(img, 3, axis=2)[0]

  img = np.split(img, [75,225], axis = 0)[1]
  char = np.split(img, [48,148,248,352,452,552], axis=1)
  char=np.array([char[1],char[2],char[4],char[5]])
  
  return np.array(char)

images =  np.array([proccess_and_split(img) for img in images_raw])

X_dataset = np.concatenate(images, axis=0)
# plt.imshow(X_dataset[0][:,:,0])
# plt.show()
# plt.imshow(X_dataset[1][:,:,0])
# plt.show()
# plt.imshow(X_dataset[2][:,:,0])
# plt.show()
# plt.imshow(X_dataset[3][:,:,0])
# plt.show()
Y_dataset = np.array([data for data in np.concatenate((labels), axis=0)])
# Split data.
print(X_dataset.shape)
data_amount = labels.shape[0]

train_ratio = .8
val_ratio = .2
test_ratio = 1 - val_ratio - train_ratio

train_amount = int(data_amount*train_ratio)
val_amount = int(data_amount*val_ratio)
test_amount = int(data_amount*test_ratio)

"""## Train CNN"""

from keras import layers
from keras import models
from keras import optimizers

from keras.utils import plot_model
from keras import backend

"""## Reset weights
Function for reinitializing the model parameters.
"""

def reset_weights(model):
    session = backend.get_session()
    for layer in model.layers: 
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel.initializer.run(session=session)

"""## Model definition"""

conv_model = models.Sequential()
conv_model.add(layers.Conv2D(32, (3, 3), activation='relu',
                             input_shape=(150, 100, 3)))
conv_model.add(layers.MaxPooling2D((2, 2)))
conv_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
conv_model.add(layers.MaxPooling2D((2, 2)))
conv_model.add(layers.Conv2D(128, (3, 3), activation='relu'))
conv_model.add(layers.MaxPooling2D((2, 2)))
conv_model.add(layers.Conv2D(128, (3, 3), activation='relu'))
conv_model.add(layers.MaxPooling2D((2, 2)))
conv_model.add(layers.Flatten())
conv_model.add(layers.Dropout(0.5))
conv_model.add(layers.Dense(512, activation='relu'))
conv_model.add(layers.Dense(36, activation='softmax'))
   
conv_model.summary()

LEARNING_RATE = 1e-4

conv_model.compile(loss='categorical_crossentropy',
                   optimizer=optimizers.RMSprop(lr=LEARNING_RATE),
                   metrics=['acc'])

history_conv = conv_model.fit(X_dataset, Y_dataset, 
                              validation_split=val_ratio, 
                              epochs=20, 
                              batch_size=16) 

plt.plot(history_conv.history['loss'])
plt.plot(history_conv.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train loss', 'val loss'], loc='upper left')
plt.show()

plt.plot(history_conv.history['acc'])
plt.plot(history_conv.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy (%)')
plt.xlabel('epoch')
plt.legend(['train accuracy', 'val accuracy'], loc='upper left')
plt.show()

"""## Test model"""

# # Display images in the training data set. 
# def displayImage(index):
#   img = X_dataset[index]
  
#   img_aug = np.expand_dims(img, axis=0)
#   y_predict = conv_model.predict(img_aug)[0]
  
#   plt.imshow(img)  
#   caption = ("                  Cat | Dog\n"+
#              "GND truth: {:.2} | {:.2}\nPredicted: {:.2} | {:.2}".
#              format(Y_dataset[index][0], Y_dataset[index][1], y_predict[0], y_predict[1]))
#   plt.text(0.5, 0.5, caption, 
#            color='orange', fontsize = 16,
#            horizontalalignment='left', verticalalignment='bottom')


# # interact(displayImage, 
# #         index=ipywidgets.IntSlider(min=0, max=X_dataset_orig.shape[0],
# #                                    step=1, value=10))
# displayImage(3)


# confusion matrix
# from: https://androidkt.com/keras-confusion-matrix-in-tensorboard/

# use scikit instead for ease of use (do confusion matrix. it's more better)

import tensorflow as tf
import pandas as pd
import seaborn as sns

y_pred = conv_model.predict_classes(X_dataset)
sess = tf.Session()


con_mat = tf.math.confusion_matrix(labels=y_labels.T, predictions=y_pred).eval(session=sess)

con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)

classes = [i for i in range(36)]

con_mat_df = pd.DataFrame(con_mat_norm,
                     index = classes, 
                     columns = classes)

figure = plt.figure(figsize=(8, 8))
sns.heatmap(con_mat_df, annot=True,cmap=plt.cm.Blues)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()