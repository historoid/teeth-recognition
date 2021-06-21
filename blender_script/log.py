import bpy
import os
import itertools

# rendering setting
res = 512
ext = '.png'
# path setting
path = '/Documents/res/09_DeepLearning/img/Lo/'
res_path = os.environ['HOME'] + path + str(res) + '_g/'

# default setting
dent = [11, 12, 13, 14, 15, 16, 17,
        21, 22, 23, 24, 25, 26, 27,
        31, 32, 33, 34, 35, 36, 37,
        41, 42, 43, 44, 45, 46, 47]
num_dent = len(dent)
ini_layer = [False] * 20
ini_layer[0] = True

# Move selected teeth to layer[1]
def move_to_layer1(TEETH):
    for i in range(len(TEETH)):
        a = bpy.data.objects[str(TEETH[i])]
        bpy.context.scene.layers[1] = True
        for k in range(20):
            a.layers[k] = (k == 1)


# Return selected teeth to layer[0]
def move_to_layer2(TEETH):
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
    move_to_layer1(TEETH)
    rendering(TEETH)
    move_to_layer2(TEETH)


# Let's Start!!
for i in range(len(dent)+1):
    x = list(itertools.combinations(dent, i))
    for k in x:
        total(k)
