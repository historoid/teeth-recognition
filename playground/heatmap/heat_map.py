# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
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

# +
import os
import keras
from keras import layers
from keras import models
from encode import DataSet
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.layers.convolutional import Conv2D, MaxPooling2D, SeparableConv2D

img_path = '/Users/seino/Documents/40_research/80_DeepLearning/playground/00_img/one'
height, width, channels = 32, 32, 3
num_classes = 32

ds = DataSet(img_path)
b = ds.make_digit()
c = ds.k_hot_encode()
x_train, x_test, y_train, y_test = ds.like_mnist(b,c, 0.2)

# +
model = models.Sequential()
model.add(layers.SeparableConv2D(32, 3, activation='relu', input_shape=(height, width, channels)))
model.add(layers.SeparableConv2D(64, 3, activation='relu'))
model.add(layers.Dropout(0.3))
model.add(layers.MaxPooling2D(2))

model.add(layers.SeparableConv2D(64, 3, activation='relu'))
model.add(layers.SeparableConv2D(128, 3, activation='relu'))
model.add(layers.Dropout(0.3))
model.add(layers.MaxPooling2D(2))

model.add(layers.SeparableConv2D(64, 3, activation='relu'))
model.add(layers.SeparableConv2D(128, 3, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.GlobalAveragePooling2D())

model.add(layers.Dense(500, activation='relu'))
model.add(layers.Dense(num_classes, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy',metrics=['acc'])
model.summary()
# -

callbacks_list = [
    keras.callbacks.EarlyStopping(
        monitor='val_acc',
        patience=1
    ),
    keras.callbacks.ModelCheckpoint(
        filepath='callbacks.h5',
        monitor='val_acc',
        save_best_only=True,
    ),
    keras.callbacks.TensorBoard(
        log_dir='./log',
        histogram_freq=1,
        batch_size=32,
        write_graph=True,
        write_grads=False,
        embeddings_freq=1,
        embeddings_layer_names=None,
        embeddings_metadata=None
    )
]

history = model.fit(x_train, y_train, batch_size=50, epochs=10, validation_data = (x_test, y_test), verbose=1)

model.save('teeth_recognition.h5')
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['train_acc', 'val_acc'], loc='lower right')
plt.show()

plt.imshow(x_train[0])

layer_outputs = [layer.output for layer in model.layers[:8]]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(x_train)
first_layer_activation = activations[0]
print(first_layer_activation.shape)

plt.matshow(first_layer_activation[0, :, :, 3], cmap='viridis')
plt.show()

plt.matshow(first_layer_activation[0, :, :, 30], cmap='viridis')
plt.show()

# +
# 各レイヤーの名前を抽出
layer_names = []
for layer in model.layers[:4]:
    layer_names.append(layer.name)

# 一行に表示する画像の数
images_per_row = 16

for layer_name, layer_activation in zip(layer_names, activations):
    # レイヤーの特徴マップ数を取得
    n_features = layer_activation.shape[-1]
    # 特徴マップのサイズを取得
    size = layer_activation.shape[1]
    # 画像の行数を計算し画像を並べる0行列を生成
    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))

    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0, :, :, col * images_per_row + row]
            # 特徴マップを標準化
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            # 平均128、標準偏差64に変換
            channel_image *= 64
            channel_image += 128
            # 0-255に符号なし8ビット整数化
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            # 所定の位置にイメージを埋め込む
            display_grid[col * size : (col + 1) * size, row * size: (row +1) * size ] = channel_image
        # 画像の表示
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')

plt.show()

# +
from keras.applications.vgg16 import VGG16

model_p = VGG16(weights='imagenet')

from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions

sample_path ='/Users/seino/Documents/res/09_DeepLearning/img/With_Gingiva/Mix/512/11,12,13,14,15,16,17,21,22,23,24,25,26,27.png'
sample = image.load_img(sample_path, target_size=(224, 224))
c = image.img_to_array(sample)
c = np.expand_dims(c, axis=0)
c = preprocess_input(c)

preds = model_p.predict(c)
print('Predicted:', decode_predictions(preds, top=3)[0])
np.argmax(preds[0])

# +
from keras import backend as K

dent_output = model_p.output[:, 643]
last_conv_layer = model_p.get_layer('block5_conv3')

grads = K.gradients(dent_output, last_conv_layer.output)[0]
pooled_grads = K.mean(grads, axis=(0, 1, 2))

iterate = K.function([model_p.input], [pooled_grads, last_conv_layer.output[0]])

pooled_grads_value, conv_layer_output_value = iterate([c])

for i in range(32):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

heatmap = np.mean(conv_layer_output_value, axis =-1)
print(heatmap)

heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
plt.matshow(heatmap)
# -

import cv2
img = cv2.imread(sample_path)
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = heatmap * 0.4  + img
cv2.imwrite('/Users/seino/Downloads/test.jpg', superimposed_img)

# +
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions

sample_path ='/Users/seino/Documents/res/09_DeepLearning/img/With_Gingiva/Mix/512/11,12,13,14,15,16,17,21,22,23,24,25,26,27.png'
sample = image.load_img(sample_path, target_size=(32, 32))
c = image.img_to_array(sample)
c = np.expand_dims(c, axis=0)
c = preprocess_input(c)
preds = model.predict(c)
np.argmax(preds[0])
from keras import backend as K

dent_output = model.output[:, 22]
last_conv_layer = model.get_layer('separable_conv2d_10')

grads = K.gradients(dent_output, last_conv_layer.output)[0]
pooled_grads = K.mean(grads, axis=(0, 1, 2))

iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])

pooled_grads_value, conv_layer_output_value = iterate([c])

for i in range(32):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

heatmap = np.mean(conv_layer_output_value, axis =-1)
print(heatmap)

heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
plt.matshow(heatmap)

# +
import os
image_path ='/Users/seino/Documents/res/09_DeepLearning/img/With_Gingiva/Mix/512/'

fn_list = os.listdir(image_path)
try:
    fn_list.remove('.DS_Store')
except:
    pass

img_fullpath_list =[]
for i in fn_list:
    img_fullpath = os.path.join(image_path, i)
    img_fullpath_list.append(img_fullpath)


# +
#How many images do you want?
img_num = 100


for k in range(img_num):
    sample_image = image.load_img(img_fullpath_list[k], target_size=(32,32))
    sample_image = image.img_to_array(sample_image)
    sample_image = np.expand_dims(sample_image, axis=0)
    sample_image = preprocess_input(sample_image)
    preds = model.predict(sample_image)
    preds_num = np.argmax(preds[0])
    
    output_dent = model.output[:, preds_num]
    layer_name = 'separable_conv2d_10'
    conv_layer_last = model.get_layer(layer_name)

    grads = K.gradients(output_dent, conv_layer_last.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([model.input], [pooled_grads, conv_layer_last.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([sample_image])

    for i in range(32):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
        heatmap = np.mean(conv_layer_output_value, axis =-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)

    img = cv2.imread(img_fullpath_list[k])
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4  + img
    cv2.imwrite('/Users/seino/Downloads/heatmap/' + str(layer_name) + '/with_gingiva/' + str(fn_list[k]) + '_heatmap.jpg', superimposed_img)
# -


