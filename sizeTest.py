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

def main():
    # imageResizeTest()
    checkBetweenEyesTest()

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
def checkBetweenEyesTest():
    detector = MTCNN()

    largeImage = cv2.cvtColor(cv2.imread(
        "./ffhq-dataset/size_128128/00000.png"), cv2.COLOR_BGR2RGB)
    result=detector.detect_faces(largeImage)
    keypoints = result[0]['keypoints']

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
        result = detector.detect_faces(smallImage)
        if not result:
            print("can'ttttttttttttttttt find face on the image")
            print('{},{}'.format(smallWidth, smallHeight))
            break
        else:
            print("can find face on the image")
            largeImage = smallImage
            largeWidth = smallWidth
            largeHeight = smallHeight
            print('{},{}'.format(smallWidth, smallHeight))
            keypoints = result[0]['keypoints']
            confidence = result[0]['confidence']
            distance = helper.getDistanceBetween2Point(keypoints['right_eye'], keypoints['left_eye'])
            print('distance : {}, confidence : {}'.format(distance,confidence))


# def outputImage(result):
#     # Result is an array with all the bounding boxes detected. We know that for 'ivan.jpg' there is only one.
#     bounding_box = result[0]['box']
#     keypoints = result[0]['keypoints']

#     cv2.rectangle(image,
#                 (bounding_box[0], bounding_box[1]),
#                 (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
#                 (0,155,255),
#                 2)

#     cv2.circle(image,(keypoints['left_eye']), 2, (0,155,255), 2)
#     cv2.circle(image,(keypoints['right_eye']), 2, (0,155,255), 2)
#     cv2.circle(image,(keypoints['nose']), 2, (0,155,255), 2)
#     cv2.circle(image,(keypoints['mouth_left']), 2, (0,155,255), 2)
#     cv2.circle(image,(keypoints['mouth_right']), 2, (0,155,255), 2)

#     cv2.imwrite("00000_drawn.jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    main()
