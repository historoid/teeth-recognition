import os
import bpy
import numpy as np


def main():
    # 初期設定の入力
    # n = int(input('生成する画像の枚数を入力して下さい。\n'))
    # res = int(input('生成する画像の解像度を入力して下さい。\n'))
    # dst = input('画像の保存先フォルダを指定して下さい。\n')
    
    n = 1000
    res = 512
    dst = 'img'
    

    train_path = dst + '/training_img'
    validation_path = dst + '/validation_img'
    image_extension = '.png'
    
    camera_close = ('Front', 'Right', 'Left')
    camera_occlusion = ('Up_Occ', 'Low_Occ')
    
    bpy.context.scene.render.resolution_x = \
    bpy.context.scene.render.resolution_y = res


    teeth_numbering = (11, 12, 13, 14, 15, 16, 17, \
                       21, 22, 23, 24, 25, 26, 27, \
                       31, 32, 33, 34, 35, 36, 37, \
                       41, 42, 43, 44, 45, 46, 47)
    teeth_numbering = np.array(teeth_numbering)

    random_selection_arr = np.random.rand(n, 28)
    random_selection_arr = np.where(random_selection_arr > 0.5, 1, 0)
    
    upper_teeth = np.intersect1d(teeth_numbering[:13], random_selection_arr)
    upper_teeth = np.append(upper_teeth, 'Maxilla')
    lower_teeth = np.intersect1d(teeth_numbering[13:], random_selection_arr)
    lower_teeth = np.append(lower_teeth, ['Mandible','Tongue'])
    
    # 生成する画像は80%が訓練、20%がテスト
    train_num = round(n*0.8)
        
    for cam in camera_close:
        
        for dent in random_selection_arr[:train_num]:
            dn, fn = make_dir_name_and_fn(train_path, dent)
            initialize_position(teeth_numbering)
            move_to_render_layer(teeth_numbering, dent)
            render_image(cam)
            save_rendering_image(cam, dn, fn, image_extension)
            render_compositing_image(cam)
            save_composite_image(cam, dn, fn, image_extension)         
        
        for dent in random_selection_arr[train_num:]:
            dn, fn = make_dir_name_and_fn(validation_path, dent)
            initialize_position(teeth_numbering)
            move_to_render_layer(teeth_numbering, dent)
            render_image(cam)
            save_rendering_image(cam, dn, fn, image_extension)
            render_compositing_image(cam)
            save_composite_image(cam, dn, fn, image_extension)
   

    for cam in camera_occlusion:
        
        for dent in random_selection_arr[:train_num]:
            dn, fn = make_dir_name_and_fn(train_path, dent)
            initialize_position(teeth_numbering)
            move_to_render_layer(teeth_numbering, dent)
            x = dent * teeth_numbering
            up_dent = np.delete(x, np.where(x==0), axis=0)
            up_dent = np.delete(up_dent, np.where(up_dent<30), axis=0)
            up_dent = np.append(up_dent, 'Maxilla')
            y = dent * teeth_numbering
            lo_dent = np.delete(y, np.where(y==0), axis=0)
            lo_dent = np.delete(lo_dent, np.where(lo_dent>30), axis=0)
            lo_dent = np.append(lo_dent, ['Mandible', 'Tongue'])
            move_up(up_dent)
            move_down(lo_dent)
            render_image(cam)
            save_rendering_image(cam, dn, fn, image_extension)
            render_compositing_image(cam)
            save_composite_image(cam, dn, fn, image_extension)
            move_up(lo_dent)
            move_down(up_dent)
        
        for dent in random_selection_arr[train_num:]:
            dn, fn = make_dir_name_and_fn(validation_path, dent)
            initialize_position(teeth_numbering)
            move_to_render_layer(teeth_numbering, dent)
            x = dent * teeth_numbering
            up_dent = np.delete(x, np.where(x==0), axis=0)
            up_dent = np.delete(up_dent, np.where(up_dent<30), axis=0)
            up_dent = np.append(up_dent, 'Maxilla')
            y = dent * teeth_numbering
            lo_dent = np.delete(y, np.where(y==0), axis=0)
            lo_dent = np.delete(lo_dent, np.where(lo_dent>30), axis=0)
            lo_dent = np.append(lo_dent, ['Mandible', 'Tongue'])
            move_up(up_dent)
            move_down(lo_dent)
            render_image(cam)
            save_rendering_image(cam, dn, fn, image_extension)
            render_compositing_image(cam)
            save_composite_image(cam, dn, fn, image_extension)
            move_up(lo_dent)
            move_down(up_dent)


# フォルダ名とファイル名の作成
def make_dir_name_and_fn(save_dir, dent_arr):
    fn = ''.join(map(str, dent_arr))
    dn = os.path.join(save_dir, fn)
    os.makedirs(dn, exist_ok=True)
    return dn, fn
    

# すべての歯を非レンダリングレイヤーに移す
def initialize_position(teeth_numbering):
    for i in teeth_numbering:
        bpy.data.objects[str(i)].hide_render = True


# 選んだ歯をレンダリングレイヤーに移す
def move_to_render_layer(teeth_numbering, dent_arr):
    for n, i in zip(teeth_numbering, dent_arr):
        bpy.data.objects[str(n)].hide_render = (not i)
        

# 選んだ歯を上方に移動させる
def move_up(selected_teeth):
    if selected_teeth.size == 0:
        pass
    else:
        for i in selected_teeth:
            bpy.data.objects[str(i)].location[2] += 6.0

            
# 選んだ歯を下方に移動させる
def move_down(selected_teeth):
    if selected_teeth.size == 0:
        pass
    else:
        for i in selected_teeth:
            bpy.data.objects[str(i)].location[2] -= 6.0

            
# レンダリング
def render_image(camera):
    bpy.context.scene.camera = bpy.data.objects[camera]
    bpy.context.scene.render.use_compositing = False
    bpy.ops.render.render(scene='1')

    
# コンポジットレンダリング
def render_compositing_image(camera):
    bpy.context.scene.camera = bpy.data.objects[camera]
    bpy.context.scene.render.use_compositing = True
    bpy.ops.render.render(scene='1')
    
    
# レンダリング画像の保存
def save_rendering_image(cam, dir_name, file_name, image_extension):
    file_name = file_name + '-' + cam + image_extension
    total_path = os.path.join(dir_name, file_name)
    bpy.data.images['Render Result'].save_render(total_path)
    

# コンポジットレンダリング画像の保存
def save_composite_image(cam, dir_name, file_name, image_extension):
    file_name = 'ss-' + file_name + '-' + cam + image_extension
    total_path = os.path.join(dir_name, file_name)
    bpy.data.images['Render Result'].save_render(total_path)
        

if __name__=='__main__':
    main()