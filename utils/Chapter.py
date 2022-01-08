import sys
sys.path.append('../')
import cv2
import glob
import numpy as np
from utils.Page import Page
from google_trans_new.google_trans_new import google_translator
from fpdf import FPDF
from yolov4.models import Yolov4
from PIL import Image
import easyocr


class Chapter:

    def __init__(self, folder="chapter/", folder_dest="chapter_trad/", extension=".jpg", lang_src="ja", lang_dst="en", name_dest="translated_chapter", imgs=None):
        """

        :param folder:
        :param folder_dest:
        :param extension:
        :param lang_src: "ja", "ch_sim", "ko", "en"
        :param lang_dst: "ja", "ch_sim", "ko", "en"
        :param name_dest: name of the translated file to be saved
        """

        self.folder = folder
        self.folder_dest = folder_dest
        self.translator = google_translator()
        self.extension = extension
        self.pages = list()
        self.name_dest = name_dest
        self.model = Yolov4(weight_path='./model/yolov4.weights', class_name_path='./model/coco_classes.txt')
        #reader = easyocr.Reader([lang_src])
        if imgs is None:
            for file_name in sorted(glob.glob(self.folder + '*' + extension)):
                self.pages.append(Page(cv2.imread(file_name), self.translator, self.model, lang_src[0], lang_dst))
        else:
            for img in imgs:
                self.pages.append(Page(img, self.translator, self.model, lang_src[0], lang_dst))

    def translate(self):
        for k in self.pages:
            k.detect_boxes()
            k.translate()

    def save_pages(self, to_pdf=False):
        if not to_pdf:
            for k in range(len(self.pages)):
                self.pages[k].save(k, self.folder_dest, self.extension)
        else:
            pages = []
            for page in self.pages:
                pages.append(Image.fromarray(page.img_final))
            pages[0].save(self.folder_dest + self.name_dest + ".pdf", "PDF" ,resolution=100.0, save_all=True,
                          append_images=pages[1:])

