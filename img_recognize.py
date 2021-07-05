import cv2


OP_MATCH_PATH = 'data/out/op_result.png'
MY_MATCH_PATH = 'data/out/my_result.png'


class ImageRecognition:
    """画像認識に関するクラス"""
    def __init__(self, img, temp, save_path, limit=0.5):
        """コンストラクタ"""
        self.img = cv2.imread(img)
        self.templ = cv2.imread(temp)
        self.save_path = save_path
        self.limit = limit

    def img_recognize(self):
        # 画像処理
        img = cv2.threshold(self.img, 190, 255, cv2.THRESH_BINARY)[1]
        templ = cv2.threshold(self.templ, 190, 255, cv2.THRESH_BINARY)[1]

        # テンプレートマッチング
        result = cv2.matchTemplate(img, templ, cv2.TM_CCOEFF_NORMED)

        # 最も類似度が高い位置を取得
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        print(f"max value: {max_val}, position: {max_loc}")

        # 描画・トリミング・保存
        if max_val >= self.limit:
            tl = max_loc[0], max_loc[1]
            br = max_loc[0] + self.templ.shape[1], max_loc[1] + self.templ.shape[0] - 30
            trimmed_img = self.img[tl[1]:br[1], tl[0]:br[0]]

            cv2.imwrite(self.save_path, trimmed_img)
            return True
        else:
            return False


if __name__ == '__main__':
    # ir = ImageRecognition('data/img/on_battle_3.png', 'data/img/op_pk_name.png', OP_MATCH_PATH)
    ir = ImageRecognition('data/img/on_battle_3.png', 'data/img/my_pk_name.png', MY_MATCH_PATH)
    ret = ir.img_recognize()
