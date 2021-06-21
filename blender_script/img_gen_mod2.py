import os
import bpy
import numpy as np

# Setting
N = 10
ext = '.png'
output_path = 'img/'
dent = (
    11, 12, 13, 14, 15, 16, 17,
    21, 22, 23, 24, 25, 26, 27,
    31, 32, 33, 34, 35, 36, 37,
    41, 42, 43, 44, 45, 46, 47
    )
cam_names = (
    'Mid_F', 'Mid_R', 'Mid_L', 'Mid_R2', 'Mid_L2', 
    'Up_F', 'Up_R', 'Up_L', 'Up_R2', 'Up_L2',
    'Lo_F', 'Lo_R', 'Lo_L', 'Lo_R2', 'Lo_L2'
    )

# make df of image list
f = lambda x: 1 if x > 0.5 else 0
dent_vec = np.frompyfunc(f, 1, 1)(np.random.rand(N, 28))

# convert vector to boolean
tof = lambda x: True if x==1 else False
dent_bool = np.frompyfunc(tof, 1, 1)(dent_vec)

# file name
for row in dent_vec:
    fn = ''.join(map(str, row))

# initialize
for i in dent:
    bpy.data.objects[str(i)].hide_render = True


for each_sample in dent_bool:
    save_path = output_path + fn
    os.makedirs(save_path, exist_ok=True)
    for each_dent, each_bool in zip(dent, each_sample):
        bpy.data.objects[str(each_dent)].hide_render = not each_bool
    for cam in cam_names:
        bpy.context.scene.camera = bpy.data.objects[cam]
        bpy.context.scene.render.use_compositing = False
        dst = save_path + '/' + fn + '-' + cam
        bpy.ops.render.render(scene='1')
        bpy.data.images['Render Result'].save_render(dst+ext)
        bpy.context.scene.render.use_compositing = True
        dst = dst + '-SS'
        bpy.ops.render.render(scene='1')
        bpy.data.images['Render Result'].save_render(dst+ext)
    for each_dent, each_bool in zip(dent, each_sampl):
        bpy.data.objects[str(each_dent)].hide_render = each_bool