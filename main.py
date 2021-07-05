import PySimpleGUI as sg
import pandas as pd
import cv2
from img_recognize import ImageRecognition
import os
import pyocr.builders
from PIL import Image
import decimal
import threading
from my_audio import Audio


# region 定数
TESSERACT_PATH = 'C:\\Program Files\\Tesseract-OCR'
POKE_DATA_CSV_PATH = 'data/poke_data.csv'
OP_POKE_NAME_PATH = 'data/img/op_pk_name.png'
MY_POKE_NAME_PATH = 'data/img/my_pk_name.png'
CURRENT_PATH = 'data/out/current.png'
OP_MATCH_PATH = 'data/out/op_result.png'
MY_MATCH_PATH = 'data/out/my_result.png'

FONT_SIZE = 16

# endregion

"""
PythonによるGUI作成
テンプレートマッチングさせた画像から文字認識
"""


# region メイン画面
class MainDisplay:
    """メイン画面"""
    def __init__(self):
        sg.theme('Default1')

        # region パーツ設定

        # region 色調調整
        button_exit = sg.Button('Exit', size=(10, 1))

        radio_none = sg.Radio('None', 'Radio', True, size=(10, 1))
        radio_threshold = sg.Radio('threshold', 'Radio', size=(10, 1), key='-THRESH-')
        radio_canny = sg.Radio('canny', 'Radio', size=(10, 1), key='-CANNY-')
        radio_blur = sg.Radio('blur', 'Radio', size=(10, 1), key='-BLUR-')
        radio_hue = sg.Radio('hue', 'Radio', size=(10, 1), key='-HUE-')
        radio_enhance = sg.Radio('enhance', 'Radio', size=(10, 1), key='-ENHANCE-')

        slider_threshold = sg.Slider((0, 255), 128, 1, orientation='h', size=(40, 15), key='-THRESH SLIDER-')
        slider_canny_a = sg.Slider((0, 255), 128, 1, orientation='h', size=(20, 15), key='-CANNY SLIDER A-')
        slider_canny_b = sg.Slider((0, 255), 128, 1, orientation='h', size=(20, 15), key='-CANNY SLIDER B-')
        slider_blur = sg.Slider((1, 100), 1, 1, orientation='h', size=(40, 15), key='-BLUR SLIDER-')
        slider_hue = sg.Slider((0, 225), 0, 1, orientation='h', size=(40, 15), key='-HUE SLIDER-')
        slider_enhance = sg.Slider((1, 255), 128, 1, orientation='h', size=(40, 15), key='-ENHANCE SLIDER-')

        frame_radio = sg.Frame('adjustment', [
          [radio_none],
          [radio_threshold, slider_threshold],
          [radio_canny, slider_canny_a, slider_canny_b],
          [radio_blur, slider_blur],
          [radio_hue, slider_hue],
          [radio_enhance, slider_enhance],
          [button_exit]
        ])

        tab_adjustment = sg.Tab('色調調整', [[sg.Text('画面効果・画面設定', size=(60, 1))], [frame_radio]])

        # endregion

        # region ポケモン情報(敵、味方)

        text_op_pk_name = sg.Text('フシギダネ　　　　　　　', font=('メイリオ', FONT_SIZE), key='-OP_NAME-')
        text_op_pk_type1 = sg.Text('くさ　　　　', font=('メイリオ', FONT_SIZE), key='-OP_TYPE1-')
        text_op_pk_type2 = sg.Text('どく　　　　', font=('メイリオ', FONT_SIZE), key='-OP_TYPE2-')
        text_op_pk_skill = sg.Text('しんりょく (ようりょくそ)　　　　　　　　　',
                                   font=('メイリオ', FONT_SIZE - 2), key='-OP_SKILL-')
        text_op_pk_hp = sg.Text('HP： 45', font=('メイリオ', FONT_SIZE), key='-OP_HP-')
        text_op_pk_atk = sg.Text('攻撃： 49', font=('メイリオ', FONT_SIZE), key='-OP_ATK-')
        text_op_pk_def = sg.Text('防御： 49', font=('メイリオ', FONT_SIZE), key='-OP_DEF-')
        text_op_pk_sp_atk = sg.Text('特攻： 65', font=('メイリオ', FONT_SIZE), key='-OP_SP_ATK-')
        text_op_pk_sp_def = sg.Text('特防： 65', font=('メイリオ', FONT_SIZE), key='-OP_SP_DEF-')
        text_op_pk_spd = sg.Text('素早さ： 45', font=('メイリオ', FONT_SIZE), key='-OP_SPD-')
        op_ev252_corr, op_ev252, op_no_corr = calc_pokemon_spd(45)
        text_op_pk_spd_2 = sg.Text(f'(Lv50)最速： {op_ev252_corr} / 準速： {op_ev252} / 無振： {op_no_corr}    ',
                                   font=('メイリオ', FONT_SIZE - 3), key='-OP_SPD_2-')

        button_op_change = sg.Button('Change', size=(10, 1), key='-OP_CHANGE-')

        text_my_pk_name = sg.Text('フシギダネ　　　　　　　', font=('メイリオ', FONT_SIZE), key='-MY_NAME-')
        text_my_pk_type1 = sg.Text('くさ　　　　', font=('メイリオ', FONT_SIZE), key='-MY_TYPE1-')
        text_my_pk_type2 = sg.Text('どく　　　　', font=('メイリオ', FONT_SIZE), key='-MY_TYPE2-')
        text_my_pk_skill = sg.Text('しんりょく (ようりょくそ)　　　　　　　　　',
                                   font=('メイリオ', FONT_SIZE - 2), key='-MY_SKILL-')
        text_my_pk_hp = sg.Text('HP： 45', font=('メイリオ', FONT_SIZE), key='-MY_HP-')
        text_my_pk_atk = sg.Text('攻撃： 49', font=('メイリオ', FONT_SIZE), key='-MY_ATK-')
        text_my_pk_def = sg.Text('防御： 49', font=('メイリオ', FONT_SIZE), key='-MY_DEF-')
        text_my_pk_sp_atk = sg.Text('特攻： 65', font=('メイリオ', FONT_SIZE), key='-MY_SP_ATK-')
        text_my_pk_sp_def = sg.Text('特防： 65', font=('メイリオ', FONT_SIZE), key='-MY_SP_DEF-')
        text_my_pk_spd = sg.Text('素早さ： 45', font=('メイリオ', FONT_SIZE), key='-MY_SPD-')
        my_ev252_corr, my_ev252, my_no_corr = calc_pokemon_spd(45)
        text_my_pk_spd_2 = sg.Text(f'(Lv50)最速： {my_ev252_corr} / 準速： {my_ev252} / 無振： {my_no_corr}    ',
                                   font=('メイリオ', FONT_SIZE - 3), key='-MY_SPD_2-')

        button_my_change = sg.Button('Change', size=(10, 1), key='-MY_CHANGE-')

        frame_op_poke = sg.Frame('opponent pokemon', [
            [text_op_pk_name],
            [text_op_pk_type1, text_op_pk_type2],
            [text_op_pk_skill],
            [text_op_pk_hp],
            [text_op_pk_atk],
            [text_op_pk_def],
            [text_op_pk_sp_atk],
            [text_op_pk_sp_def],
            [text_op_pk_spd],
            [text_op_pk_spd_2],
            [button_op_change]
        ])

        frame_my_poke = sg.Frame('my pokemon', [
            [text_my_pk_name],
            [text_my_pk_type1, text_my_pk_type2],
            [text_my_pk_skill],
            [text_my_pk_hp],
            [text_my_pk_atk],
            [text_my_pk_def],
            [text_my_pk_sp_atk],
            [text_my_pk_sp_def],
            [text_my_pk_spd],
            [text_my_pk_spd_2],
            [button_my_change]
        ])

        tab_pokemon = sg.Tab('ポケモン', [
                                        [sg.Text('情報', size=(60, 1))],
                                        [frame_op_poke],
                                        [frame_my_poke]
                                        ])

        # endregion

        # endregion

        # レイアウト設定
        self.layout = [[sg.TabGroup([[tab_pokemon, tab_adjustment]], tab_location='topleft')]]

        # 動画設定
        self.cap = cv2.VideoCapture('pk_test.mp4')  # 動画ファイル指定
        # self.cap = cv2.VideoCapture(0)  # キャプボ

        # ウィンドウ生成
        self.window = sg.Window('POKE_ACT', self.layout, location=(1300, 10))

    def main(self):
        """イベントループ"""
        while True:
            event, values = self.window.read(timeout=5)

            # フレームの画像が読み込めたかどうかを示すbool値と、画像の配列ndarrayのタプルを受け取る
            ret_frame, frame = self.cap.read()
            if not ret_frame:
                break

            # リサイズ
            frame = cv2.resize(frame, dsize=(1280, 720))

            # Exit押下あるいは閉じるボタン押下で終了
            if event == 'Exit' or event == sg.WIN_CLOSED:
                break

            # 敵ポケモンの値を変更する時
            if event == '-OP_CHANGE-':
                # 画像読み込み
                cv2.imwrite(CURRENT_PATH, frame)
                # テンプレートマッチング
                ir = ImageRecognition(CURRENT_PATH, OP_POKE_NAME_PATH, OP_MATCH_PATH)
                ret_img = ir.img_recognize()
                # マッチしていればトリミングした画像から文字認識
                if ret_img:
                    result_sentence = get_pokemon_name_by_ocr(OP_MATCH_PATH)

                    # ポケモンの特定
                    d_pk = ''
                    for idx in df_pokemon.index:
                        if idx in result_sentence[:6]:
                            d_pk = idx

                    # ポケモン入力
                    if d_pk:
                        self.window['-OP_NAME-'].update(d_pk)
                        self.window['-OP_TYPE1-'].update(df_pokemon.loc[d_pk, 'タイプ1'])
                        self.window['-OP_TYPE2-'].update(df_pokemon.loc[d_pk, 'タイプ2'])
                        self.window['-OP_SKILL-'].update('特性：' + df_pokemon.loc[d_pk, '特性(隠し特性)'])
                        self.window['-OP_HP-'].update('HP：' + str(df_pokemon.loc[d_pk, 'HP']))
                        self.window['-OP_ATK-'].update('攻撃：' + str(df_pokemon.loc[d_pk, '攻撃']))
                        self.window['-OP_DEF-'].update('防御：' + str(df_pokemon.loc[d_pk, '防御']))
                        self.window['-OP_SP_ATK-'].update('特攻：' + str(df_pokemon.loc[d_pk, '特攻']))
                        self.window['-OP_SP_DEF-'].update('特防：' + str(df_pokemon.loc[d_pk, '特防']))
                        self.window['-OP_SPD-'].update('素早さ：' + str(df_pokemon.loc[d_pk, '素早さ']))
                        op_ev252_corr, op_ev252, op_no_corr = calc_pokemon_spd(df_pokemon.loc[d_pk, '素早さ'])
                        self.window['-OP_SPD_2-']\
                            .update(f'(Lv50)最速： {op_ev252_corr} / 準速： {op_ev252} / 無振： {op_no_corr}')

            # 味方ポケモンの値を変更する時
            if event == '-MY_CHANGE-':
                # 画像読み込み
                cv2.imwrite(CURRENT_PATH, frame)
                # テンプレートマッチング
                ir = ImageRecognition(CURRENT_PATH, MY_POKE_NAME_PATH, MY_MATCH_PATH)
                ret_img = ir.img_recognize()

                if ret_img:
                    result_sentence = get_pokemon_name_by_ocr(MY_MATCH_PATH)

                    # ポケモンの特定
                    d_pk = ''
                    for idx in df_pokemon.index:
                        if idx in result_sentence[:6]:
                            d_pk = idx

                    # ポケモン入力
                    if d_pk:
                        self.window['-MY_NAME-'].update(d_pk)
                        self.window['-MY_TYPE1-'].update(df_pokemon.loc[d_pk, 'タイプ1'])
                        self.window['-MY_TYPE2-'].update(df_pokemon.loc[d_pk, 'タイプ2'])
                        self.window['-MY_SKILL-'].update('特性：' + df_pokemon.loc[d_pk, '特性(隠し特性)'])
                        self.window['-MY_HP-'].update('HP：' + str(df_pokemon.loc[d_pk, 'HP']))
                        self.window['-MY_ATK-'].update('攻撃：' + str(df_pokemon.loc[d_pk, '攻撃']))
                        self.window['-MY_DEF-'].update('防御：' + str(df_pokemon.loc[d_pk, '防御']))
                        self.window['-MY_SP_ATK-'].update('特攻：' + str(df_pokemon.loc[d_pk, '特攻']))
                        self.window['-MY_SP_DEF-'].update('特防：' + str(df_pokemon.loc[d_pk, '特防']))
                        self.window['-MY_SPD-'].update('素早さ：' + str(df_pokemon.loc[d_pk, '素早さ']))
                        my_ev252_corr, my_ev252, my_no_corr = calc_pokemon_spd(df_pokemon.loc[d_pk, '素早さ'])
                        self.window['-MY_SPD_2-']\
                            .update(f'(Lv50)最速： {my_ev252_corr} / 準速： {my_ev252} / 無振： {my_no_corr}')

            if values['-THRESH-']:
                # cv2.cvtColor⇒RGBやBGR、HSVなど様々な色空間を相互に変換できる
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)[:, :, 0]
                # cv2.threshold⇒しきい値より大きければある値(白)を割り当て，そうでなければ別の値(黒)を割り当てる
                frame = cv2.threshold(frame, values['-THRESH SLIDER-'], 255, cv2.THRESH_BINARY)[1]
            elif values['-CANNY-']:
                # cv2.Canny⇒エッジ検出。引数は、入力画像、minVal、maxVal
                frame = cv2.Canny(frame, values['-CANNY SLIDER A-'], values['-CANNY SLIDER B-'])
            elif values['-BLUR-']:
                # cv2.GaussianBlur⇒引数は、入力画像、カーネルサイズ、ガウス分布のシグマx
                frame = cv2.GaussianBlur(frame, (21, 21), values['-BLUR SLIDER-'])
            elif values['-HUE-']:
                # cv2.COLOR_BGR2HSV⇒BGR⇒HSV変換
                # HSVとは、色相(Hue)、彩度(Saturation・Chroma)、明度(Value・Brightness)の三つの成分からなる色空間のこと
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                frame[:, :, 1] += int(values['-HUE SLIDER-'])
                frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
            elif values['-ENHANCE-']:
                enh_val = values['-ENHANCE SLIDER-'] / 40
                clahe = cv2.createCLAHE(clipLimit=enh_val, tileGridSize=(8, 8))
                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

            cv2.imshow('Video', frame)
            # imgbytes = cv2.imencode('.png', frame)[1].tobytes()
            # window['-IMAGE-'].update(data=imgbytes)

        self.window.close()
        self.cap.release()
        cv2.destroyAllWindows()
# endregion


def read_pokemon_data(path):
    """
    名前がindexのDataFrameを作成
    :param path: csvのファイルパス
    :return: データフレーム
    """
    df = pd.read_csv(path, index_col=1)
    df = df[['HP', '攻撃', '防御', '特攻', '特防', '素早さ', 'タイプ1', 'タイプ2', '特性(隠し特性)']]
    return df


def get_pokemon_name_by_ocr(path):
    # OCRエンジンの取得
    tools = pyocr.get_available_tools()
    tool = tools[0]
    # 画像の読み込み
    img_org = Image.open(path)
    # OCRの実行
    builder = pyocr.builders.TextBuilder()
    result_sentence = tool.image_to_string(img_org, lang="jpn", builder=builder)

    return result_sentence


def calc_pokemon_spd(poke_spd):
    """
    素早さ種族値を元にLv50時の最速/準速/無振りの値を返す
    :param poke_spd: 素早さ種族値
    :return: Lv50時の最速/準速/無振りの値
    """
    ev252_corr = decimal.Decimal(str((poke_spd + 52) * decimal.Decimal('1.1'))).quantize(decimal.Decimal('0'),
                                                                                         rounding=decimal.ROUND_DOWN)
    ev252 = poke_spd + 52
    no_corr = poke_spd + 20
    return ev252_corr, ev252, no_corr


def play_audio():
    a = Audio('pk_test.mp4')
    a.play_sound()


if __name__ == '__main__':
    # Tesseractのパスを通す
    if TESSERACT_PATH not in os.environ["PATH"].split(os.pathsep):
        os.environ["PATH"] += os.pathsep + TESSERACT_PATH
    # csv読み込み
    df_pokemon = read_pokemon_data(POKE_DATA_CSV_PATH)
    # GUI表示
    display_1 = MainDisplay()
    display_1.main()
    # thread1 = threading.Thread(target=display_1.main)
    # thread2 = threading.Thread(target=play_audio)
    # thread1.start()
    # thread2.start()
    # thread1.join()
    # thread2.join()


