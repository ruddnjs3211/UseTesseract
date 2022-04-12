import speech_recognition as sr
import os
from pydub import AudioSegment
from gtts import gTTS
from PIL import Image, ImageDraw, ImageFont
import textwrap
import urllib3
import base64
import json


def txtToMp3(name, text):
    print("txtToMp3")
    i = True
    fileName = name + '.mp3'  # 음성 파일을 저장하는 경로와 파이명 지정
    while i:
        try:
            tts = gTTS(text=text, lang='ko')
            tts.save(fileName)

            return name
        except:
            os.remove(fileName)
            i = False
            return False

def mp3Towav(name):
    # AudioSegment 라이브러리로 음성포멧 조정
    print("mp3Towav")

    w = AudioSegment.from_mp3(name + '.mp3')
    w.export(name + '.wav', format='wav')
    return name + '.wav'

def Voice_Recognition(fileName):  # 음성 인식하는 함수 선언
    # fileName = "hello.wav"
    print("Voice_Recognition : "+ fileName)

    r = sr.Recognizer()
    with sr.AudioFile(fileName) as source:

        audio = r.listen(source)

        said = " "

        try:
            said = r.recognize_google(audio_data=audio, language='ko-KR')
            print(said)
        except Exception as e:
            print("Exception: " + str(e))

    os.remove(fileName)  # 생성했던 wav 삭제
    return said


def max_judge(arg):
    max_v = arg[0]
    index = 0

    for i in range(1, len(arg)):
        if arg[i] > max_v:
            max_v = arg[i]
            index = i
    # print(arg)
    return index


def textToImage(message, name):
    # Image size
    W = 640
    H = 640
    # bg_color = 'rgb(214, 230, 245)'  # 아이소프트존
    bg_color = 'rgb(255, 255, 255)'

    # font setting
    font = ImageFont.truetype('C:/Windows/Fonts/batang.ttf', size=12)
    font_color = 'rgb(0, 0, 0)'  # or just 'black'
    # 원래 윈도우에 설치된 폰트는 아래와 같이 사용 가능하나,
    # 아무리 해도 한글 폰트가 사용이 안 되어.. 같은 폴더에 다운받아 놓고 사용함.
    # font = ImageFont.truetype("arial.ttf", size=28)

    image = Image.new('RGB', (W, H), color=bg_color)
    draw = ImageDraw.Draw(image)

    # Text wraper to handle long text
    # 40자를 넘어갈 경우 여러 줄로 나눔
    lines = textwrap.wrap(message, width=40)

    # start position for text
    x_text = 50
    y_text = 50

    # 각 줄의 내용을 적음
    for line in lines:
        width, height = font.getsize(line)
        draw.text((x_text, y_text), line, font=font, fill=font_color)
        y_text += height
        # height는 글씨의 높이로, 한 줄 적고 나서 height만큼 아래에 다음 줄을 적음

    # 안에 적은 내용을 파일 이름으로 저장
    returnName = '{}.png'.format(name+"M")
    image.save(returnName)

    return returnName

def imageToPdf(fileName, wantName):
    image1 = Image.open(fileName)
    im1 = image1.convert('RGB')
    im1.save(wantName+'.pdf')
    os.remove(fileName)  # 생성했던 png 삭제


def Voice_Recognition_2(fileName):  # 음성 인식하는 함수 선언
    openApiURL = "http://aiopen.etri.re.kr:8000/WiseASR/Recognition"
    accessKey = "7bab25fb-47a1-4e03-97d3-8daeb0bd6fc8"
    audioFilePath = fileName
    languageCode = "korean"

    file = open(audioFilePath, "rb")
    audioContents = base64.b64encode(file.read()).decode("utf8")
    file.close()

    requestJson = {
        "access_key": accessKey,
        "argument": {
            "language_code": languageCode,
            "audio": audioContents
        }
    }

    http = urllib3.PoolManager()
    response = http.request(
        "POST",
        openApiURL,
        headers={"Content-Type": "application/json; charset=UTF-8"},
        body=json.dumps(requestJson)
    )

    print("[responseCode] " + str(response.status))
    print("[responBody]")
    print("===== 결과 확인 ====")
    data = json.loads(response.data.decode("utf-8", errors='ignore'))
    print(data['return_object']['recognized'])

    os.remove(fileName)  # 생성했던 wav 삭제
    return data['return_object']['recognized']
