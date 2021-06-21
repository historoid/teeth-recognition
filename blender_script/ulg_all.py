import os
import bpy
import itertools

# rendering setting
res = 32
ext = '.png'
# path setting
path = '/Documents/40_research/80_DeepLearning/playground/00_img/'
res_path = os.environ['HOME'] + path + str(res) + '/'

# default setting
dent = [11, 12, 13, 14, 15, 16, 17,
        21, 22, 23, 24, 25, 26, 27,
        31, 32, 33, 34, 35, 36, 37,
        41, 42, 43, 44, 45, 46, 47]
num_dent = len(dent)


def move_to_layer1(TEETH):
    for i in range(len(TEETH)):
        for k in range(20):
            bpy.data.objects[str(TEETH[i])].layers[k] = (k == 1)


def move_to_layer0(TEETH):
    for i in range(len(TEETH)):
        a = bpy.data.objects[str(TEETH[i])]
        for k in range(20):
            a.layers[k] = (k == 0)


def rendering(TEETH):
    bpy.ops.render.render()
    maped_teeth = map(str, TEETH)
    file_name = ','.join(maped_teeth)
    savepath = res_path + file_name + ext
    bpy.data.images['Render Result'].save_render(savepath)


def total(TEETH):
    move_to_layer1(TEETH)
    rendering(TEETH)
    move_to_layer0(TEETH)


# Let's Start!!
for i in range(len(dent)+1):
    x = list(itertools.combinations(dent, i))
    for k in x:
        total(k)
