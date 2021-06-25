import pathlib

class GetFilePath(self, img_main_dir):
    def __init__(self):
        img_sub_dir = pathlib.Path(img_main_dir).glob('[0,1]*/')
        self.label = [np.array(i.name) for i in img_sub_dir]
        # 通常画像のパス
        norm_img_path = [pathlib.Path(img_main_dir).glob('*/' + str(i) + '*.png')for i in labels]
        # セグメンテーション画像のパス
        sgmt_img_path = [pathlib.Path(img_main_dir).glob('*/ss-' + str(i) + '*.png')for i in labels]
        