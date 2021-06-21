import bpy
import os
import itertools

# rendering setting

ext = '.png'
path = '/Documents/res/09_DeepLearning/img//Teeth_Only/Up/'
bpy.context.scene.render.resolution_percentage = 100

# default setting
dent = [11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 26, 27, 28]
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
res = [32, 64, 128, 256, 512]
for r in res:
    res_path = os.environ['HOME'] + path + str(r) + '/'
    bpy.context.scene.render.resolution_x = r
    bpy.context.scene.render.resolution_y = r
    for i in range(len(dent)+1):
        x = list(itertools.combinations(dent, i))
        for k in x:
            total(k)
print('All combinations were made.')