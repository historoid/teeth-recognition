import os
import numpy as np
from keras.preprocessing.image import img_to_array, load_img
from sklearn.model_selection import train_test_split

class DataSet:

    def __init__(self, path):
        self.path = path
        self.img_list = os.listdir(self.path)
        try:
            self.img_list.remove('.DS_Store')
        except:
            pass
        self.x = []
        self.y = []
        self.dent = [
            11, 12, 13, 14, 15, 16, 17, 18,
            21, 22, 23, 24, 25, 26, 27, 28,
            31, 32, 33, 34, 35, 36, 37, 38,
            41, 42, 43, 44, 45, 46, 47, 48]

    def make_digit(self):
        for fn in self.img_list:
            dst = os.path.join(self.path, fn)
            t = img_to_array(load_img(dst))
            self.x.append(t)
        self.x = np.asarray(self.x)
        self.x = self.x.astype('float32')
        self.x = self.x/255.0
        return self.x

    def k_hot_encode(self):
        for i in self.img_list:
            teeth = [0]*32
            re = i.strip('[').strip('.png').strip(']')
            mo = re.split(',')
            tl = [int(k) for k in mo]
            for p in tl:
                if p in self.dent:
                    m = self.dent.index(p)
                    teeth[m] = 1
            self.y.append(teeth)
        self.y = np.asarray(self.y)
        return self.y

    def like_mnist(self, x, y, z):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=z)
        return x_train, x_test, y_train, y_test
