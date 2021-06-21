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

img_path = '/Users/seino/Documents/res/09_DeepLearning/playground/dataset/one'
height, width, channels = 32, 32, 3
num_classes = 32

ds = DataSet(img_path)
b = ds.make_digit()
c = ds.k_hot_encode()
x_train, x_test, y_train, y_test = ds.like_mnist(b,c, 0.2)


model = models.Sequential()
model.add(layers.SeparableConv2D(36, 3, activation='relu', input_shape=(height, width, channels)))
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

history = model.fit(x_train, y_train, batch_size=100, epochs=10, validation_data = (x_test, y_test), verbose = 1)
model.save('teeth.h5')
