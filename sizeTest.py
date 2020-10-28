#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# MIT License
#
# Copyright (c) 2019 Iván de Paz Centeno
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import cv2
from mtcnn import MTCNN
import helper
import numpy as np
import pandas as pd

df = pd.DataFrame(columns=['optimalImageSize',
                           'optimalDistanceBetweenEyes', 'optimalConfidence'])

def main():
    # imageResizeTest()

    for i in range(0,1000):
        if 0<=i and i<10:
            fileName=format(i,'05')
        elif 10<=i and i<100:
            fileName=format(i,'05')
        else:
            fileName=format(i,'05')
        fileName = fileName+'.png'
        checkBetweenEyesTest(fileName)
    # checkBetweenEyesTest('00993.png')

    df.to_csv("output.csv",mode='w')

#image 사이즈 자체를 1/2하면서 언제부터 detoector.detect_faces()가 face를 탐색할 수 없는지 Test
# 128*128에서 시작하여, 32*32까지는 정상 탐색됐으나, 16*16에서는 탐색되지 않았다.
#하나의 척도가 될 수 있겠으나, 이미지 별로 그 안에서 face가 차지하는 크기가 다르기 때문에 좋은 기준이 아닌듯.
#눈과 눈 사이 거리를 새로운 기준으로 잡고 다시 해보자.
# def imageResizeTest():
#     detector = MTCNN()

#     largeImage = cv2.cvtColor(cv2.imread(
#         "./ffhq-dataset/size_128128/00000.png"), cv2.COLOR_BGR2RGB)
#     largeImageSize = largeImage.shape
#     largeWidth = largeImageSize[0]
#     largeHeight = largeImageSize[1]

#     while (1):
#         smallWidth = int(largeWidth/2)
#         smallHeight = int(largeHeight/2)

#         smallImage = cv2.resize(src=largeImage, dsize=(
#             smallWidth, smallHeight), interpolation=cv2.INTER_AREA)
#         result = detector.detect_faces(smallImage)
#         if not result:
#             print("can'ttttttttttttttttt find face on the image")
#             print('{},{}'.format(smallWidth, smallHeight))
#             break
#         else:
#             print("can find face on the image")
#             largeImage = smallImage
#             largeWidth = smallWidth
#             largeHeight = smallHeight

# 눈과 눈 사이의 거리를 살펴봄.
# 줄이는 간격은 여전히 사진 전체 사이즈를 1/2함.
# image 32*32, 눈 사이 거리 8.06225774829855 일 때까지 정상적으로 face dection함.

def checkBetweenEyesTest(fileName):
    detector = MTCNN()
    largeImage = cv2.cvtColor(cv2.imread(
        './ffhq-dataset/size_128128/'+fileName), cv2.COLOR_BGR2RGB)
    largeResult=detector.detect_faces(largeImage)
    keypoints = largeResult[0]['keypoints']

    #right_eye는 실제 오른쪽 눈이 아니라, 사진상 오른쪽 눈의 좌표임
    distance = helper.getDistanceBetween2Point(keypoints['right_eye'],keypoints['left_eye'])
    print ('first distance : {}'.format(distance))

    largeImageSize = largeImage.shape
    largeWidth = largeImageSize[0]
    largeHeight = largeImageSize[1]

    while (1):
        smallWidth = int(largeWidth/2)
        smallHeight = int(largeHeight/2)

        smallImage = cv2.resize(src=largeImage, dsize=(
            smallWidth, smallHeight), interpolation=cv2.INTER_AREA)
        smallResult = detector.detect_faces(smallImage)
        if not smallResult:
            print("can'ttttttttttttttttt find face on the image")
            print('{},{}'.format(smallWidth, smallHeight))
            findLeastDetectableDistance(smallImage, smallWidth, smallHeight, largeImage, largeResult, largeWidth, largeHeight, fileName)
            break
        else:
            print("can find face on the image")
            print('{},{}'.format(smallWidth, smallHeight))
            largeImage = smallImage
            largeResult = smallResult
            largeWidth = smallWidth
            largeHeight = smallHeight
            keypoints = smallResult[0]['keypoints']
            confidence = smallResult[0]['confidence']
            distance = helper.getDistanceBetween2Point(keypoints['right_eye'], keypoints['left_eye'])
            print('distance : {}, confidence : {}'.format(distance,confidence))

#use binary search to find the least detectable distance
def findLeastDetectableDistance(image, smallWidth, smallHeight, largeIamge, largeResult,largeWidth, largeHeight, fileName):
    detector = MTCNN()
    print('find the least detectable distance\n')
    undetectedSizeList = []#detect를 해보았으나, 결과가 output되지 않는 size의 list
    localLeastImage=largeIamge
    localLeastResult=largeResult
    detected = 0

    while (1):
        middleWidth = int((largeWidth+smallWidth)/2)
        middleHeight = int((largeHeight+smallHeight)/2)

        #face detection을 할 수 있는 가장 작은 이미지 탐색 완료.
        #이렇게 해주긴했는데, 00000.jpg에서 이거 없이 돌리면 size 22에서 처음에는 안 되다가, 이미지 조정하던 중에 confidence 0.7~로 됨. 이거 추후에 해결해야함.
        #->이거 우선 (middleHeight == largeHeight) or (middleHeight == smallHeight)이걸로 해결함
        if (middleHeight in undetectedSizeList) or (middleHeight == largeHeight) or (middleHeight == smallHeight):
            print("size {} is in undetectedSizeList".format(middleHeight))
            outputImage(localLeastImage, localLeastResult,fileName)
            appendToDf(fileName, largeHeight,localLeastResult)
            break

        if not detected:
            #이미지 확대 ->  바이큐빅 보간법
            middleImage = cv2.resize(src=image, dsize=(
                middleWidth, middleHeight), interpolation=cv2.INTER_CUBIC)
        else: 
            #이미지 축소 -> 영역 보간법
            middleImage = cv2.resize(src=image, dsize=(
                middleWidth, middleHeight), interpolation=cv2.INTER_AREA)
        
        result = detector.detect_faces(middleImage)

        if not result:
            print("can'ttttttttttttttttt find face on the image")
            print('{},{}\n'.format(middleWidth, middleHeight))
            undetectedSizeList.append(middleHeight)
            smallWidth = middleWidth
            smallHeight = middleHeight
            detected = 0

        else:
            print("can find face on the image")
            image = middleImage
            largeWidth = middleWidth
            largeHeight = middleHeight
            detected = 1
            print('{},{}'.format(middleWidth, middleHeight))
            # printImageInfo(result)
            #python은 pass by assignment(Call by obejct reference)임.
            #image객체는 numpy.ndarray로 mutable이므로 여기서 image에 rectangle이랑 circle을 씌워주면, 다시 findLeastDetectableDistance()로 돌아가서도 해당 것들이 씌워진 상태로 함수를 돎.
            #그래서 정상적으로 탐색이 안 됨. 따라서 image를 tmpImage에 deep copy해줌.
            #여기는 중간중간 가능한 이미지 전부 output할 때만 사용.
            localLeastImage=image.copy()
            localLeastResult=result.copy()
            # outputImage(tmpImage, localLeatResult,~~~~)

def outputImage(image,result,fileName):
    bounding_box = result[0]['box']
    keypoints = result[0]['keypoints']

    cv2.rectangle(image,
                (bounding_box[0], bounding_box[1]),
                (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
                (0,155,255),
                2)

    cv2.circle(image,(keypoints['left_eye']), 2, (0,155,255), 2)
    cv2.circle(image,(keypoints['right_eye']), 2, (0,155,255), 2)
    cv2.circle(image,(keypoints['nose']), 2, (0,155,255), 2)
    cv2.circle(image,(keypoints['mouth_left']), 2, (0,155,255), 2)
    cv2.circle(image,(keypoints['mouth_right']), 2, (0,155,255), 2)

    outputFileName =fileName+"_drawn.jpg"
    print(cv2)
    # print("output")
    cv2.imwrite('./output/'+outputFileName, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))    

def printImageInfo(result):
    keypoints = result[0]['keypoints']
    confidence = result[0]['confidence']
    distance = helper.getDistanceBetween2Point(
        keypoints['right_eye'], keypoints['left_eye'])
    print('distance : {}, confidence : {}\n'.format(distance, confidence))

def appendToDf(fileName, largeHeight, localLeastResult):
    infoList=[]

    fileName=fileName.split('.')[0]
    infoList.append(largeHeight)
    keypoints = localLeastResult[0]['keypoints']
    localDistance = helper.getDistanceBetween2Point(keypoints['right_eye'], keypoints['left_eye'])
    infoList.append(localDistance)
    infoList.append(localLeastResult[0]['confidence'])
    df.loc[fileName]=infoList
    # print (df)

if __name__ == "__main__":
    main()
