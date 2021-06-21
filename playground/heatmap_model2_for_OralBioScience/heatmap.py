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

# 画像のファイル名から正解ラベルを作る
# まずはモジュールのインポート
import re
import random
import pathlib
import numpy as np
import keras
from keras.preprocessing.image import img_to_array, load_img
from sklearn.model_selection import train_test_split


class makeDataSet:
    # インスタンス変数の定義とファイル名の取得
    def __init__(self, img_dir, img_size):
        self.img_size = img_size
        self.img_dir_obj = pathlib.Path(img_dir)
        self.img_all_list = list(self.img_dir_obj.glob('*.png'))
        self.img_all_list = random.sample(self.img_all_list, len(self.img_all_list))
        self.x = []
        self.y = []
        self.dent = [
            11, 12, 13, 14, 15, 16, 17, 18,
            21, 22, 23, 24, 25, 26, 27, 28,
            31, 32, 33, 34, 35, 36, 37, 38,
            41, 42, 43, 44, 45, 46, 47, 48]


    # 画像をnp配列に変換
    def imageToArray(self, img_list):
        for fn in img_list:
            img_array = img_to_array(load_img(fn))
            self.x.append(img_array)
        self.x = np.asarray(self.x, dtype=np.float32)
        self.x /= 255.0
        return self.x


    # ファイル名をk-hotエンコーディング
    def k_hot_encode(self, img_list):
        for fn in img_list:
            k_hot_teeth = [0]*32
            fn = re.sub('.*-', '', str(fn))
            fn = fn.strip('.png')
            fn = fn.split(',')
            fn = [int(k) for k in fn]
            for tooth in fn:
                if tooth in self.dent:
                    k_hot_teeth[self.dent.index(tooth)] = 1
            self.y.append(k_hot_teeth)
        self.y = np.asarray(self.y, dtype=np.float32)
        return self.y


    # 使用する画像の枚数を指定
    def pickupImages(self, pick_num):
        self.pickup_img = self.img_all_list[:pick_num]
        return self.pickup_img
    
    
    # 画像サイズを変更したいとき
    def adjustedImgArray(self, img_list):
        for fn in img_list:
            ad_img = keras.preprocessing.image.load_img(fn, target_size=(self.img_size, self.img_size))
            ad_img_array = img_to_array(ad_img)
            self.x.append(ad_img_array)
        self.x = np.asarray(self.x, dtype=np.float32)
        self.x /= 255.0    
        return self.x

    
    #  訓練データと確認データを切り分ける
    def devideData(self, img_array, img_label, test_size):
        x_train, x_test, y_train, y_test = train_test_split(img_array, img_label, test_size=test_size)
        return x_train, x_test, y_train, y_test

# +
img_dir = '/Users/seino/Documents/research/DeepLearning/playground/img'
ds = makeDataSet(img_dir, 32)
'''
img_list = ds.pickupImages(1000)
img_array = ds.adjustedImgArray(img_list)
img_label = ds.k_hot_encode(img_list)
'''
img_list = ds.img_all_list
img_array = ds.adjustedImgArray(img_list)
img_label = ds.k_hot_encode(img_list)

test_size = 0.2
x_train, x_test, y_train, y_test = ds.devideData(img_array, img_label, test_size)

# +
# 可視化して確認
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# %matplotlib inline

print(x_train.shape)
print(y_train.shape)
print(y_train[0])
plt.imshow(x_train[0])
plt.show()
# -

# 機械学習に必要なモジュールをインポートする
import keras
from keras import layers
from keras import models
from keras.utils import plot_model
from keras.models import load_model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.layers.convolutional import Conv2D, MaxPooling2D, SeparableConv2D

# 画像のサイズ定義
img_size = ds.img_size
height, width, channels = img_size, img_size, 3
class_num = len(ds.dent) # 歯種がクラスになる

# +
# モデルの構築
model = models.Sequential()
model.add(layers.SeparableConv2D(img_size, 3, activation='relu', input_shape=(height, width, channels)))
model.add(layers.SeparableConv2D(img_size*2, 3, activation='relu'))
model.add(layers.Dropout(0.3))
model.add(layers.MaxPooling2D(2))

model.add(layers.SeparableConv2D(img_size*2, 3, activation='relu'))
model.add(layers.SeparableConv2D(img_size*4, 3, activation='relu'))
model.add(layers.Dropout(0.3))
model.add(layers.MaxPooling2D(2))

model.add(layers.SeparableConv2D(img_size*4, 3, activation='relu'))
model.add(layers.SeparableConv2D(img_size*8, 3, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.GlobalAveragePooling2D())

model.add(layers.Dense(img_size*4, activation='relu'))
model.add(layers.Dense(img_size*2, activation='relu'))
model.add(layers.Dense(class_num, activation='sigmoid'))

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

history = model.fit(
    x_train, y_train, batch_size=50,
    epochs=20, validation_data = (x_test, y_test), verbose=1
)

# +
model.save('heatmap.h5')

train_acc = history.history['acc']
val_acc = history.history['val_acc']

train_loss = history.history['loss']
val_loss = history.history['val_loss']

plt.plot(train_acc, linestyle='--', color='b')
plt.plot(val_acc, linestyle='-', color='#e46409')
plt.title('Accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['train_acc', 'val_acc'], loc='lower left')
plt.show()

plt.plot(train_loss, linestyle='--', color='b')
plt.plot(val_loss, linestyle='-', color='#e46409')
plt.title('Loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train_loss', 'val_loss'], loc='upper right')
plt.show()
# -

# 活性化マップを可視化する。
layer_outputs = [layer.output for layer in model.layers[:4]]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(x_train)
first_layer_activation = activations[0]
print(first_layer_activation.shape)
plt.matshow(first_layer_activation[0, :, :, 3], cmap='viridis')
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
# -

# 以下、学習モデルを用いて、ヒートマップを作成する。

# +
# 学習モデルを用いて、ヒートマップと元の画像を重ねて表示する
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

# ターゲット画像へのローカルパス
img_path = '/Users/seino/Documents/research/DeepLearning/playground/img/RU2512ULG-11,12,13,14,16,17,23,24,26,32,33,34,36,41,43,45,47.png'
# 読み込み
img = image.load_img(img_path, target_size=(ds.img_size, ds.img_size))
# 画像を配列に変換
img_na = image.img_to_array(img)
# バッチに変換するために次元を追加
img_na = np.expand_dims(img_na, axis=0)
# 前処理。チャネルごとに色を正規化
img_na = preprocess_input(img_na)
# -

# サンプル画像に対する予測
preds = model.predict_classes(img_na)
pred_class = preds[0]

# 予測ベクトルはpred_class
dent_output = model.output[:, pred_class]
# 畳み込み層を取得
last_conv_layer = model.get_layer('separable_conv2d_2')
# 特徴マップの勾配を取得
grads = keras.backend.gradients(dent_output, last_conv_layer.output)[0]
# 形状が(32,)のベクトル
# 各ラベルは特定の特徴マップチャネルの勾配の平均強度
pooled_grads = keras.backend.mean(grads, axis=(0, 1, 2))
# サンプル画像に基づいて、平均強度と特徴出力マップにアクセス
iterate = keras.backend.function([model.input], [pooled_grads, last_conv_layer.output[0]])
# 上記をnumpy配列に
pooled_grads_value, conv_layer_output_value = iterate([img_na])
# このクラスに対するチャネルの重要度を特徴マップに重み付け
for i in range(32):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
# 最終的な特徴マップのチャネルごとの平均値がクラスの活性化ヒートマップになる
heatmap = np.mean(conv_layer_output_value, axis=-1)
# ヒートマップの後処理
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
plt.matshow(heatmap)

# cv2でスーパーインポーズ
import cv2
# 元画像の読み込み
img_cv = cv2.imread(img_path)
# ヒートマップのサイズと元画像のサイズを合わせる
heatmap = cv2.resize(heatmap, (img_cv.shape[1], img_cv.shape[0]))
# ヒートマップをRGBに変換
heatmap = np.uint8(255 * heatmap)
cv2.imwrite('/Users/seino/Downloads/map.png', heatmap)
# ヒートマップを元画像に適応
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
# 透明度40%でスーパーインポーズ
superimposed_heatimg = heatmap * 0.5 + img_cv
cv2.imwrite('/Users/seino/Downloads/heat.png', superimposed_heatimg)


