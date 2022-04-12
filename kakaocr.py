from PIL import Image, ImageDraw, ImageFont
import cv2
import pytesseract
import numpy as np
from googletrans import Translator
from difflib import SequenceMatcher
import numpy
import re

i = input("파일 이름 입력 : ")
iiii = i+'.png'
print(iiii)

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
img = cv2.imread(iiii)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img2 = Image.open(iiii)
fnt = ImageFont.truetype("NanumPen.ttf", 20, encoding="UTF-8")
im = Image.open(iiii)
draw=ImageDraw.Draw(im)


# Enlarge
height, width = gray.shape
gray21 = cv2.resize(gray, (2 * width, 2 * height), interpolation=cv2.INTER_LINEAR)
gray41 = cv2.resize(gray, (4 * width, 4 * height), interpolation=cv2.INTER_LINEAR)
gray81 = cv2.resize(gray, (8 * width, 8 * height), interpolation=cv2.INTER_LINEAR)

# Denoising
gray21 = cv2.fastNlMeansDenoising(gray21, h=10, searchWindowSize=21, templateWindowSize=7)
gray41 = cv2.fastNlMeansDenoising(gray41, h=10, searchWindowSize=21, templateWindowSize=7)
gray81 = cv2.fastNlMeansDenoising(gray81, h=10, searchWindowSize=21, templateWindowSize=7)


kernel = np.ones((5,5), np.uint8)
gray21 = cv2.erode(gray21, kernel, iterations=1)
gray41 = cv2.erode(gray41, kernel, iterations=1)
gray81 = cv2.erode(gray81, kernel, iterations=1)
gray21 = cv2.morphologyEx(gray21, cv2.MORPH_OPEN, kernel)
gray41 = cv2.morphologyEx(gray41, cv2.MORPH_OPEN, kernel)
gray81 = cv2.morphologyEx(gray81, cv2.MORPH_OPEN, kernel)
gray21 = cv2.erode(gray21, kernel, iterations=1)
gray41 = cv2.erode(gray41, kernel, iterations=1)
gray81 = cv2.erode(gray81, kernel, iterations=1)
gray21 = cv2.erode(gray21, kernel, iterations=1)
gray41 = cv2.erode(gray41, kernel, iterations=1)
gray81 = cv2.erode(gray81, kernel, iterations=1)



# Thresholding
gray_pin = 0
ret, gray21 = cv2.threshold(gray21, gray_pin, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
gray22 = cv2.adaptiveThreshold(gray21,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
ret, gray41 = cv2.threshold(gray41, gray_pin, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
gray42 = cv2.adaptiveThreshold(gray41,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
ret, gray81 = cv2.threshold(gray81, gray_pin, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
gray82 = cv2.adaptiveThreshold(gray81,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)

# inverting
gray21[0:2 * height] = ~gray21[0:2 * height]
gray22[0:2 * height] = ~gray22[0:2 * height]
gray41[0:4 * height] = ~gray41[0:4 * height]
gray42[0:4 * height] = ~gray42[0:4 * height]
gray81[0:8 * height] = ~gray81[0:8 * height]
gray82[0:8 * height] = ~gray82[0:8 * height]

gray21 = cv2.medianBlur(gray21, 5)
gray22 = cv2.medianBlur(gray22, 5)
gray41 = cv2.medianBlur(gray41, 5)
gray42 = cv2.medianBlur(gray42, 5)
gray81 = cv2.medianBlur(gray81, 5)
gray82 = cv2.medianBlur(gray82, 5)

hImg, wImg, _ = img.shape
boxes0 = pytesseract.image_to_string(gray22, lang='kor')
boxes1 = pytesseract.image_to_string(gray21, lang='kor')
boxes2 = pytesseract.image_to_string(gray42, lang='kor')
boxes3 = pytesseract.image_to_string(gray41, lang='kor')
boxes4 = pytesseract.image_to_string(gray82, lang='kor')
boxes5 = pytesseract.image_to_string(gray81, lang='kor')

boxlist = []
boxlist2 = []
boxlist2.append(boxes0.rstrip("\n"))
boxlist2.append(boxes1.rstrip("\n"))
boxlist2.append(boxes2.rstrip("\n"))
boxlist2.append(boxes3.rstrip("\n"))
boxlist2.append(boxes4.rstrip("\n"))
boxlist2.append(boxes5.rstrip("\n"))

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
                    matchpoint = matchpoint + SequenceMatcher(None, i, j).ratio()
        point = (matchlen[index] - numpy.median(matchlen))*5
        print(index,    matchlen[index],    point/numpy.median(matchlen),  float(matchpoint) + point/numpy.median(matchlen),  float(matchpoint))
        matchlist.append(float(matchpoint) + point/numpy.median(matchlen))
        index = index +1

    return matchlist

def max_judge(arg):
    max_v = arg[0]
    index = 0

    for i in range(1, len(arg)):
        if arg[i] > max_v:
            max_v = arg[i]
            index = i
    # print(arg)
    return index

max = max_judge(match(boxlist))
print(max)
if max == 0:
    boxes = boxes0
    gray = cv2.resize(gray22, (width, height), interpolation=cv2.INTER_LINEAR)
elif max == 1:
    boxes = boxes1
    gray = cv2.resize(gray21, (width, height), interpolation=cv2.INTER_LINEAR)
elif max == 2:
    boxes = boxes2
    gray = cv2.resize(gray42, (width, height), interpolation=cv2.INTER_LINEAR)
elif max == 3:
    boxes = boxes3
    gray = cv2.resize(gray41, (width, height), interpolation=cv2.INTER_LINEAR)
elif max == 4:
    boxes = boxes4
    gray = cv2.resize(gray82, (width, height), interpolation=cv2.INTER_LINEAR)
elif max == 5:
    boxes = boxes5
    gray = cv2.resize(gray81, (width, height), interpolation=cv2.INTER_LINEAR)


if len(boxes)>0:
    boxes = boxes.replace('\n\n','\n')
    print(boxes)
    print("========== 번역 ============================================")
    # k-e번역
    translator = Translator()
    print(translator.translate(''.join(boxes)).text)
cv2.imshow('Result', gray)
cv2.waitKey(0)

# 1. tts 와 연결(마음api 사용 가능할듯함)
# 2. 문장 수정 프로그램과 연동/ 한영 모두
# 추후 오디오북 등으로 사용 가능하도록