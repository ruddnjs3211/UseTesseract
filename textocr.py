import os
from PIL import Image, ImageDraw, ImageFont
import cv2
import pytesseract
import numpy as np
from googletrans import Translator
from difflib import SequenceMatcher
import numpy
import re
import txttomp3 as txttomp3

file_list = os.listdir()
for flist in file_list:
    if flist[-3:] == "png":
        print(flist)


iiii = input("파일 이름 입력 : ")
print(iiii)

img = Image.open(iiii)
print(img.info)
if img.info['dpi'][0] < 299:
    dpi = (300, 300)
    img.save(iiii, dpi=dpi)
print(img.info['dpi'])

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
img = cv2.imread(iiii)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

hImg, wImg, _ = img.shape
boxes = pytesseract.image_to_boxes(img, lang='kor')
matchlist = []
for b in boxes.splitlines():
    b = b.split(' ')
    x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
    matchlist.append(h-y)
    cv2.rectangle(img, (x, hImg-y), (w, hImg-h), (0, 0, 255), 3)
    cv2.putText(img, b[0], (x, hImg-y),
                cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 2)

ppp = float(18/numpy.median(matchlist))


img2 = Image.open(iiii)
# fnt = ImageFont.truetype("NanumPen.ttf", 20, encoding="UTF-8")
im = Image.open(iiii)
draw = ImageDraw.Draw(im)

print(numpy.median(matchlist))
print(ppp*numpy.median(matchlist))

ppp = 4.5*ppp

print(ppp)

# Enlarge
height, width = gray.shape
gray21 = cv2.resize(gray, (int(ppp * width), int(ppp * height)),
                    interpolation=cv2.INTER_LINEAR)

# Denoising
gray21 = cv2.fastNlMeansDenoising(
    gray21, h=10, searchWindowSize=21, templateWindowSize=7)


kernel = np.ones((5, 5), np.uint8)
# gray21 = cv2.erode(gray21, kernel, iterations=1)
gray21 = cv2.morphologyEx(gray21, cv2.MORPH_OPEN, kernel)
# gray21 = cv2.erode(gray21, kernel, iterations=1)
# gray21 = cv2.erode(gray21, kernel, iterations=1)


# Thresholding
gray_pin = 0
ret, gray21 = cv2.threshold(gray21, gray_pin, 255,
                            cv2.THRESH_BINARY | cv2.THRESH_OTSU)
gray22 = cv2.adaptiveThreshold(
    gray21, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
# inverting
gray21[0:int(ppp * height)] = ~gray21[0:int(ppp * height)]
gray22[0:int(ppp * height)] = ~gray22[0:int(ppp * height)]
gray21 = cv2.medianBlur(gray21, 5)
gray22 = cv2.medianBlur(gray22, 5)
hImg, wImg, _ = img.shape
boxes0 = pytesseract.image_to_string(gray22, lang='kor')
boxes1 = pytesseract.image_to_string(gray21, lang='kor')
boxlist = []
boxlist2 = []
boxlist2.append(boxes0.rstrip("\n"))
boxlist2.append(boxes1.rstrip("\n"))
for boxes in boxlist2:
    boxlist.append(re.compile('[가-힣]+').findall(boxes))


def match(arg):
    matchlist = []
    matchlen = []
    for i in arg:
        matchlen.append(len(i))
    index = 0
    for i in arg:
        matchpoint = 0.0
        if len(i) > 0:
            for j in arg:
                if (len(j) > 0) and (SequenceMatcher(None, i, j).ratio() < 1):
                    matchpoint = matchpoint + \
                        SequenceMatcher(None, i, j).ratio()
        point = (matchlen[index] - numpy.median(matchlen))*5
        print(index,    matchlen[index],    point/numpy.median(matchlen),
              float(matchpoint) + point/numpy.median(matchlen),  float(matchpoint))
        matchlist.append(float(matchpoint) + point/numpy.median(matchlen))
        index = index + 1

    return matchlist


max = txttomp3.max_judge(boxlist2)
print(max)
if max == 0:
    boxes = boxes0
    grays = cv2.resize(gray22, (width, height), interpolation=cv2.INTER_LINEAR)
elif max == 1:
    boxes = boxes1
    grays = cv2.resize(gray21, (width, height), interpolation=cv2.INTER_LINEAR)
innertext = ''.join(boxes)
if len(boxes) > 0:
    boxes = boxes.replace('\n\n', '\n')
    print(boxes)
    # print("========== 번역 ============================================")
    # # k-e번역
    # translator = Translator()
    # print(translator.translate(innertext).text)

# hImgs, wImgs = grays.shape
# boxes = pytesseract.image_to_boxes(grays, lang='kor')
# matchlist = []
# for b in boxes.splitlines():
#     b = b.split(' ')
#     x,y,w,h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
#     matchlist.append(h-y)
#     cv2.rectangle(grays, (x,hImgs-y), (w,hImgs-h), (255,255,255), 3)
#     cv2.putText(grays,b[0], (x,hImgs-y), cv2.FONT_HERSHEY_COMPLEX,1,(0,255,255),2)

cv2.imshow('Result', grays)


# txttomp3.imageToPdf(txttomp3.textToImage(txttomp3.Voice_Recognition_2(
#     txttomp3.mp3Towav(txttomp3.txtToMp3(i, innertext))), i), i)

cv2.waitKey(0)
