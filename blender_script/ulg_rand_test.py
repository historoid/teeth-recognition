# In[]:
import bpy
import random

# In[]:
# rendering setting
img_num = 500
layer_num = 20
img_size = 512
model = 'ULG'
save_dir = '/Users/seino/Documents/research/DeepLearning/playground/img/'
DENT = [
    11, 12, 13, 14, 15, 16, 17,
    21, 22, 23, 24, 25, 26, 27,
    31, 32, 33, 34, 35, 36, 37,
    41, 42, 43, 44, 45, 46, 47
]
CAMERA = [
    'F', 'FD', 'FU',
    'L', 'LD', 'LU', 'L2', 'LD2', 'LU2',
    'R', 'RD', 'RU', 'R2', 'RD2', 'RU2'
    ]


# In[]:
def moveToLayer1(dent_list):
    for i in dent_list:
        for k in range(layer_num):
            bpy.data.objects[str(i)].layers[k] = (k == 1)


def moveToLayer0(dent_list):
    for i in dent_list:
        for k in range(layer_num):
            bpy.data.objects[str(i)].layers[k] = (k == 0)


def makeFileName(dent_list, cam_posi):
    core_name = ','.join(map(str, dent_list))
    dst = save_dir + cam_posi + str(img_size) +\
        model + '-' + core_name + '.png'
    return dst


def makeRandomTeethList():
    selection_num = random.randint(1, len(DENT))
    random_dent_list = random.sample(DENT, selection_num)
    return random_dent_list


# In[]
for cam_posi in CAMERA:
    for i in range(img_num):
        dent_list = makeRandomTeethList()
        dent_list.sort()
        moveToLayer1(dent_list)
        fn = makeFileName(dent_list, cam_posi)
        bpy.context.scene.camera = bpy.data.objects[cam_posi]
        bpy.ops.render.render()
        bpy.data.images['Render Result'].save_render(fn)
        moveToLayer0(dent_list)
