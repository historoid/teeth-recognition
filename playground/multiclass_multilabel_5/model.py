# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import re
import keras
import pathlib
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras import layers
from keras import models
from keras.utils import plot_model
from keras.models import load_model
from keras.models import Sequential
from keras.preprocessing.image import img_to_array, load_img
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.layers.convolutional import Conv2D, MaxPooling2D, SeparableConv2D


# %%
num_classes = 32
height, width, channels = 32, 32, 3
using_img_num = 1000
dent = [
    11, 12, 13, 14, 15, 16, 17, 18,
    21, 22, 23, 24, 25, 26, 27, 28,
    31, 32, 33, 34, 35, 36, 37, 38,
    41, 42, 43, 44, 45, 46, 47, 48
    ]
data_path = '/Users/seino/Documents/research/DeepLearning/playground/multiclass_multilabel_5/32'
data_obj = pathlib.Path(data_path)
all_img = list(data_obj.glob('*.png'))
using_img = random.sample(all_img, using_img_num)
x = []
y = []
for each_img in using_img:
    # 画像を数値に変換
    x_img = load_img(each_img)
    x_array = img_to_array(x_img)
    x.append(x_array)
    # ファイル名をラベルに変換
    fname = each_img.name
    fname = re.sub('.*-', '', fname)
    fname = fname.strip('.png')
    fname = fname.split(',')
    fname = [int(k) for k in fname]
    teeth = [0]*32
    for i in fname:
        indx = dent.index(i)
        teeth[indx] = 1
    y.append(teeth)
# xとyが出揃ってから処理すべきもの
x = np.asarray(x)
x = x.astype('float32')
x = x/255.0
y = np.asarray(y)
# 訓練ファイルと検証ファイルを分割
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# %%
height, width, channels = 32, 32, 3
hidden_units = 4  # 1 is baseline
input_param = 32  # 32 is baseline
dropout_rate = 0.4  # 0.1 is baseline
batch_p = 60  # 100 is baseline
epoch_p = 50  # 10 is baseline
model = models.Sequential()
model.add(layers.SeparableConv2D(
    input_param, 3, activation='relu',
    input_shape=(height, width, channels)))
model.add(layers.SeparableConv2D(input_param*2, 3, activation='relu'))
model.add(layers.MaxPooling2D(2))
model.add(layers.Dropout(dropout_rate))
model.add(layers.MaxPooling2D(2))
model.add(layers.SeparableConv2D(input_param*2, 3, activation='relu'))
model.add(layers.SeparableConv2D(input_param*4, 3, activation='relu'))
model.add(layers.Dropout(dropout_rate))
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(input_param*2, activation='relu'))
model.add(layers.Dense(num_classes, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model.summary()
model_arch_name = 'R' + str(input_param) + '-H' + str(hidden_units) + '-D'\
    + str(int(dropout_rate*100)) + '-B' + str(batch_p) + '-E' + str(epoch_p)
plot_model(model, show_shapes=True, to_file='./'+model_arch_name+'.png')

history = model.fit(
    x_train, y_train,
    batch_size=batch_p,
    epochs=epoch_p,
    validation_data=(x_test, y_test),
    verbose=1)
model.save('./'+model_arch_name+'.h5')
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title(model_arch_name)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['train_acc', 'val_acc'], loc='lower right')
plt.savefig('./acc_'+model_arch_name+'.png')
plt.show()

jpg_name = 'F64ULG-15,17,22,31,34,36,37,43,46,47'
model_file_name = 'R32-H4-D40-B60-E50.h5'

model = load_model('/Users/seino/Documents/research/DeepLearning/playground/\
    multiclass_multilabel_5/log/R32-H4-D40-B60-E50.h5')

img_path = ('/Users/seino/Documents/research/DeepLearning/playground/img/32/\
    F32ULG-15,25,34,35,37,43,44,47.png')
img = img_to_array(load_img(img_path, target_size=(32, 32)))
img_nad = img_to_array(img)/255
img_nad = img_nad[None, ...]

label = [
    11, 12, 13, 14, 15, 16, 17, 18,
    21, 22, 23, 24, 25, 26, 27, 28,
    31, 32, 33, 34, 35, 36, 37, 38,
    41, 42, 43, 44, 45, 46, 47, 48
]
pred = model.predict(img_nad, batch_size=1, verbose=0)
score = np.max(pred)
pred_label = label[np.argmax(pred[0])]
print('name:', pred_label)
print('score:', score)
plt.imshow(load_img(img_path))

print(pred_label, type(pred_label))
print(pred, type(pred))
pred_labels = label[np.argsort(pred)]
print(pred_labels, type(pred_labels))
