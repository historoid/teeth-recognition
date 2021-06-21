import os
import bpy
import itertools

# Setting
output_path = '/Users/historoid/local_data/10_Res/14_Teeh_Recognition/img'

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

for i in dent:
    bpy.data.objects[str(i)].hide_render = True

for num in range(len(dent)):
    for dent_comb in itertools.combinations(dent, num):
        fn = ','.join(map(str, dent_comb))
        save_path = output_path + fn
        os.makedirs(save_path, exist_ok=True)
        for dent_i in dent_comb:
            bpy.data.objects[str(dent_i)].hide_render = False
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
        for dent_i in dent_comb:
            bpy.data.objects[str(dent_i)].hide_render = True