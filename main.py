from utils.Chapter import Chapter
import urllib
import cv2
import numpy as np

def download_chapter(urls):
    pages = []
    for url in urls:
        hdr = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64)'}

        req = urllib.request.Request(url, headers=hdr)
        resp = urllib.request.urlopen(req)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        pages.append(cv2.imdecode(image, cv2.IMREAD_COLOR))
    return pages

def detect(pages=None, name=None):
    # Use a breakpoint in the code line below to debug your script.
    if pages is None:
        chapter = Chapter()  # Press Ctrl+F8 to toggle the breakpoint.
    else:
        if name is None:
            chapter = Chapter(imgs=pages)
        else:
            chapter = Chapter(name_dest=name, imgs=pages)
    chapter.translate()
    chapter.save_pages(True)

if __name__ == '__main__':

    detect()


