import os
import bpy
import pandas as pd
import random

# Setting
output_path = '/Users/historoid/Documents/Research/Theme/teeth_recognition/img/'

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
ext = '.png'

# make df of image list
N = 10
df_vec = pd.DataFrame()
for i in range(28):
    row = random.sample(range(0, N, 1), k=N)
    row = pd.Series(row, name=str(i))
    ser = row.map(lambda x: 0 if x <= row.median() else 1)
    df_vec = pd.concat([df_vec, ser], axis=1)


# convert vector to boolean
df_bool = df_vec.applymap(lambda x: True if x == 1 else False)

# file name
for row in df_vec.itertuples(name=None):
    fn = '-'.join(map(str, row))

# initialize
# No teeth can be seen
for i in dent:
    bpy.data.objects[str(i)].hide_render = True

# meta data
bpy.context.scene.render.use_stamp_note = True
bpy.context.scene.render.stamp_note_text = fn


#
for each_sample in df_bool.itertuples(name=None):
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