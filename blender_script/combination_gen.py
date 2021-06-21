import bpy
import os
import random
import math

# rendering setting
res = 512
ext = '.png'
# path setting
path = '/Documents/res/03_DeepLearning/img/'
res_path = os.environ['HOME'] + path + str(res) + '/'
bpy.context.scene.render.resolution_x = res
bpy.context.scene.render.resolution_y = res

# default setting
img = 50000
dent = [11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 26, 27, 28]
# dent = [31, 32, 33, 34, 35, 36, 37, 38, 41, 42, 43, 44, 45, 46, 47, 48]
num_dent = len(dent)
ini_layer = [False] * 20
ini_layer[0] = True


# Initiate
def initiation():
    for i in range(num_dent):
        a = bpy.data.objects[str(dent[i])]
        a.layers = ini_layer


def combinations(n, r):
    a = math.factorial(n)
    b = math.factorial(r)
    c = math.factorial(n - r)
    return a // (b * c)


# Move selected teeth to layer[1]
def move(TEETH):
    for i in range(len(rand)):
        a = bpy.data.objects[str(TEETH[i])]
        bpy.context.scene.layers[1] = True
        for k in range(20):
            a.layers[k] = (k == 1)


# Return selected teeth to layer[0]
def rev(TEETH):
    for i in range(len(TEETH)):
        a = bpy.data.objects[str(TEETH[i])]
        bpy.context.scene.layers[0] = True
        for k in range(20):
            a.layers[k] = (k == 0)


# Render the upper and Save images
def rendering(TEETH):
    bpy.ops.render.render()
    savepath = res_path + str(TEETH) + ext
    bpy.data.images['Render Result'].save_render(savepath)


# Total Process
def total(TEETH):
    move(TEETH)
    rendering(TEETH)
    rev(TEETH)


def trim_fname(file_name):
    if file_name == '.DS_Store':
        pass
    else:
        file_name = file_name.strip('[').strip('.png').strip(']').split(', ')
        file_name = [int(k) for k in file_name]
    return file_name


# Let's Start!!
files = os.listdir(res_path)
num_file = len(files)
teeth_list = []

for t in files:
    additional_fn = trim_fname(t)
    teeth_list.append(additional_fn)
while num_file <= img:
    num_select = random.randint(1, num_dent)
    comb = combinations(num_dent, num_select)
    if comb >= img:
        for EPOCH in range(img):
            rand = random.sample(dent, num_select)
            rand.sort()
            if rand in teeth_list:
                pass
            else:
                total(rand)
                num_file = num_file + 1
                if num_file >= img:
                    break
    else:
        for EPOCH in range(comb):
            rand = random.sample(dent, num_select)
            rand.sort()
            if rand in teeth_list:
                pass
            else:
                total(rand)
                num_file = num_file + 1
                if num_file >= img:
                    break
print('There are enough image files!')