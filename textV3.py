# -*- coding: utf8 -*-

import cv2
import numpy as np
import pytesseract
from corner import white_box
from location import click_event

# Mention the installed location of Tesseract-OCR in your system
pytesseract.pytesseract.tesseract_cmd = 'F:/c++opencv/finalProject/tesseract/tesseract.exe'
large = white_box().black_box(cv2.imread('b_center.jpg'))
text = pytesseract.image_to_string(large, lang='vie', config='--psm 1 --oem 3')
#  -c tessedit_char_whitelist=0123456789')
a = text.encode('utf-8')
f = open('recognized.txt', 'wb')
f.write(a)
f.close()
# cv2.imshow('image',large)
# cv2.setMouseCallback('image',click_event)
# cv2.waitKey(0)
