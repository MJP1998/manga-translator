import sys
sys.path.append('../')
import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pytesseract
from PIL import Image, ImageDraw, ImageFont
from pytesseract import Output
import textwrap
from model.utils import detect
import logging
import matplotlib.pyplot as plt
pytesseract.pytesseract.tesseract_cmd = r'C:/PROGRA~1/Tesseract-OCR/tesseract.exe' # r'C:/Users/mpilc/AppData/Local/Tesseract-OCR/tesseract.exe'

class Page:

    def __init__(self, img, translator, model, lang_origin, lang_dest):
        """

        :param img: image of the page
        :param translator: google translator
        :param lang_origin: language of the original page
        :param lang_dest: language of destination
        """
        self.img = img
        self.text_img = []
        self.text_bbox = []
        self.img_final = img.copy()
        self.translator = translator
        self.lang_origin = lang_origin
        self.lang_dest = lang_dest

        self.model = model
        #self.reader = reader

    def detect_boxes(self):
        """
        input :
        model Keras
        :return: detected boxes by the model
        """
        predict = self.model.predict_img(self.img, return_output=True)
        column = predict.columns.values
        for k in predict.index.values:
            if isinstance(predict[column[0]][k], np.int64):
                self.text_bbox.append([predict[column[0]][k], predict[column[1]][k], predict[column[6]][k],
                                       predict[column[7]][k]])

    def detect_text_inside(self, image, tessdata_dir=r'C:/PROGRA~1/Tesseract-OCR/tessdata/'):
        """

        :param image: part of the image detected by the model
        :param tessdata_dir: path to the tessdata directory
        :return: text extracted by tesseract
        """
        #preprocessing
        img = cv2.cvtColor(cv2.resize(image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC),
                                 cv2.COLOR_BGR2GRAY)
        kernel = np.ones((1, 1), np.uint8)
        img = cv2.dilate(img, kernel, iterations=1)
        img = cv2.erode(img, kernel, iterations=1)
        #tried = cv2.bilateralFilter(grayImage, 9, 75, 75)
        #(thresh, blackAndWhiteImage) = cv2.threshold(tried, 127, 255, cv2.THRESH_TOZERO)
        img = cv2.adaptiveThreshold(cv2.medianBlur(img, 3), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
        #change color if black background
        if cv2.countNonZero(img)/(img.shape[0]*img.shape[1]) < 0.4:
            img = cv2.bitwise_not(img)

        #Text Detection
        if self.lang_origin == "ch_sim":
            config = f"-l chi_sim+chi_sim_vert --oem 1 --psm 5 --tessdata-dir {tessdata_dir}"
        elif self.lang_origin == "ko":
            config = f"-l kor+kor_vert --oem 1 --psm 5 --tessdata-dir {tessdata_dir}"
        elif self.lang_origin == "en":
            config = f"-l eng --oem 1 --psm 5 --tessdata-dir {tessdata_dir}"
        else:
            config = f"-l jpn+jpn_vert --oem 1 --psm 5 --tessdata-dir {tessdata_dir}" #+chi_sim+kor+kor_vert +eng+chi_sim+kor+kor_vert
        text = pytesseract.image_to_data(img, output_type='data.frame', config=config)
        text = text[text.conf > 50]
        lines = text.groupby('block_num')['text'].apply(list)
        text = ""
        for line in lines:
            for txt in line:
                text += str(txt)
            text += " "

        return text
        """
        #with EasyOCR
        results = self.reader.readtext(img, detail=0)
        text = ""
        for i in range(len(results)):
            text += results[i]
        return text"""

    def putMaskOnImg(self, H, W, y1, x1, w, h):
        """
        :param H: height of image
        :param W: width of image
        :param y1: pos detected by the model
        :param x1: pos detected by the model
        :param w: width of the detection
        :param h: height of the detectio
        :return: modify self.img with the mask created => remove the text with inpainting in order to preserve the background
        """

        mask = np.zeros([H, W, 3])
        for y in range(y1 + int(0.05*h), min(y1 + int(0.95*h), H)):
            for x in range(x1 + int(0.05*w), min(x1 + int(0.95*w), W)):
                mask[y, x] = np.array([1, 1, 1])

        mask = cv2.cvtColor(np.uint8(np.array(mask)), cv2.COLOR_RGB2GRAY)
        self.img_final = cv2.inpaint(self.img_final, mask, 3, cv2.INPAINT_TELEA)

    def add_text(self, img, text, corner,
                 text_color=(0, 0, 0), text_size=20, font='NotoSansJP-Regular.otf', **option):

        new_img = img
        new_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(new_img)
        font_text = ImageFont.truetype(font=font, size=text_size, encoding=option.get('encoding', 'utf-8'))
        draw.text(corner, text, text_color, font=font_text)
        cv2_img = cv2.cvtColor(np.asarray(new_img), cv2.COLOR_RGB2BGR)
        img[:] = cv2_img[:]

    def translate(self, vertical_space=30, font_size=0.7, font_color=None, show_detected_text=False):
        if show_detected_text:
            img2 = self.img.copy()
        for k in range(len(self.text_bbox)):
            x1, y1, w, h = self.text_bbox[k]
            self.text_img.append(self.detect_text_inside(self.img[int(max(y1-2, 0)):int(min((y1+h+2),
                                                                                            self.img.shape[0])), int(max(x1-2, 0)):int(min(x1+w+2, self.img.shape[1]))]))
            if self.text_img[k] is None or self.text_img[k] == "" or self.text_img[k].isspace():
                continue
            text = self.translator.translate(self.text_img[k], lang_tgt=self.lang_dest)
            self.putMaskOnImg(self.img.shape[0], self.img.shape[1], y1, x1, w, h)

            if show_detected_text:
                img2 = cv2.rectangle(img2, (x1, y1), (x1 + w, y1 + h), (36, 255, 12), 1)
                self.cv2_img_add_text(img2, self.text_img[k], (x1, y1 - 10), text_rgb_color=(36, 255, 12), text_size=25, font='NotoSansJP-Regular.otf')

            shift = 2
            text_width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_size, 3)[0][0] / len(text)
            lines = textwrap.wrap(text, width=max(1, int((w - shift) / text_width)))  # On met le texte en plusieurs lignes délimitées par les cadres
            y2 = y1 + 15
            x2 = x1+shift
            font_color_white = None
            if font_color is None:
                moyenne = np.array(self.img[y1:y1 + h, x1:x1 + w]).mean()
                if moyenne > 200:
                    font_color = (0, 0, 0, 255)
                elif moyenne < 100:
                    font_color_white = (255, 255, 255, 255)
                else:
                    font_color_white = (255, 150, 150, 255)
            for line in lines:
                # TODO: center the text, give it as much space as possible, adapt the font size
                self.img_final = cv2.putText(
                        self.img_final,  # numpy array on which text is written
                        line,  # text
                        (x2, y2),  # position at which writing has to start
                        cv2.FONT_HERSHEY_SIMPLEX,  # font family
                        font_size,  # font size
                        font_color if font_color is not None else font_color_white,
                        3)  #
                y2 += vertical_space
        if show_detected_text:
            plt.imshow(img2)
            plt.show()

    def save(self, k, folder_dest, ext):
        cv2.imwrite(folder_dest + '/' + str(k) + ext, self.img_final)
