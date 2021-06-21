import bpy
import random

# rendering setting
res = 512
img_num = 50000
camera = [
    'Front', 'FrontDown', 'FrontUp',
    'Left1', 'Left1Down', 'Left1Up',
    'Left2', 'Lef21Down', 'Left2Up',
    'Right1', 'Right1Down', 'Right1Up',
    'Right2', 'Right2Down', 'Right2Up',
]
model = 'ULG'
p = '/Users/seino/Downloads/'
save_p = p + str(res) + '/'
bpy.context.scene.render.resolution_x = res
bpy.context.scene.render.resolution_y = res

dent = [
    11, 12, 13, 14, 15, 16, 17,
    21, 22, 23, 24, 25, 26, 27,
    31, 32, 33, 34, 35, 36, 37,
    41, 42, 43, 44, 45, 46, 47
]


def img_gen(TEETH):
    # move to layer1 (rendering layer)
    for i in TEETH:
        for k in range(20):
            bpy.data.objects[str(i)].layers[k] = (k == 1)
    # rendering and save the image
    bpy.ops.render.render()
    dst = ','.join(map(str, TEETH))
    dst = save_p + cam_posi + str(res) + model + '-' + dst + '.png'
    bpy.data.images['Render Result'].save_render(dst)
    # back to layer0
    for i in TEETH:
        for k in range(20):
            bpy.data.objects[str(i)].layers[k] = (k == 0)


for p in range(img_num):
    t = random.sample(dent, random.randint(1, len(dent)))
    t.sort()
    img_gen(t)
