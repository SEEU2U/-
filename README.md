# 영상 객체 인식 및 검출
>MediaPipe와 KNN 을 사용하여 OpenCV로 인식 및 검출

# MediaPipe란
> MediaPipe는 Google에서 제공하는 AI Framework입니다. 비디오형식 데이터를 이용한 다양한 비전 AI 기능을 파이프라인 형태로 손쉽게 사용할 수 있도록 제공되는 프레임워크입니다.

# KNN의 의미
>KNN은 간단하지만 높은 정확성을 가져 흔히 사용되는 분류 알고리즘입니다.
>
>새로운 데이터가 들어왔을 때 학습 받은 데이터들과 비교하여 예측하고 분류하는 알고리즘 입니다.

# 객체 인식이란
> 객체 인식은 이미지 또는 영상 상의 객체를 식별하는 컴퓨터 비전 기술입니다.
> 객체 인식은 딥러닝과 머신 러닝 알고리즘을 통해 산출되는 핵심 기술입니다.
![스크린샷 2024-05-28 160233](https://github.com/SEEU2U/RPS-Machine/assets/162940944/db80d768-ddbf-4bf0-bff3-d24c70980356)

## 코드 설명
### 가져오기 및 초기설정
```bash
import tkinter as tk
from tkinter import messagebox
from PIl import Image, ImageTk
import cv2
import mediapipe as mp
import numpy as np
import os
import time

- 'tkinter'와 'messagebox' : GUI 창을 만들고 메시지 박스를 표시하기 위해 사용함.
- 'PIL'의 'Image'와 'ImageTk' : 이미지를 열고 Tkinter에서 표시하기 위해 사용함.
- 'cv2' : OpenCV 라이브러리, 영상 처리를 위해 사용함.
- 'mediapipe' : 손 제스처 인식을 위한 라이브러리임.
- 'numpy' : 수학 연산을 위한 라이브러리임.
- 'os' : 파일 경로를 다루기 위해 사용함.
- 'time' : 시간 측정을 위해 사용함.

max_num_hands = 2
gesture = {
    0: 'fist', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five',
    6: 'six', 7: 'rock', 8: 'spiderman', 9: 'yeah', 10: 'ok',
}
rps_gesture = {0: 'rock', 5: 'paper', 9: 'scissors'}

- 'max_num_hands' : 인식할 최대 손 개수임.
- 'gesture' : 손 제스처의 레이블임.
- 'rps_gesture' : 가위바위보 게임에 사용되는 제스처임.

## MediaPipe 설정
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=max_num_hands,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

- MediaPipe Hands 솔루션과 그리기 유틸리티를 초기화함.
- 손 인식을 위한 'Hands' 객체를 생성, 인식과 추적의 최소 신뢰도를 설정함.

## KNN 모델 로드
base_path = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_path, 'data/gesture_train.csv')
file = np.genfromtxt(file_path, delimiter=',')
angle = file[:, :-1].astype(np.float32)
label = file[:, -1].astype(np.float32)
knn = cv2.ml.KNearest_create()
knn.train(angle, cv2.ml.ROW_SAMPLE, label)

- 현재 스크립트의 디렉토리를 기준으로 제스처 데이터를 로드함.
- 'numpy'를 사용하여 CSV 파일에서 데이터를 불러와서 각도와 레이블로 분리함.
- KNN 모델을 초기화하고 훈련함.

## 전역 변수 초기화
leftHand_wins = 0
rightHand_wins = 0
win_time = time.time()
win_limit_sec = 3
time_remaining = win_limit_sec

- 게임에서 왼손과 오른손의 승리 횟수를 초기화함
- 최근 승리 시간을 기록하고, 승리 판정 제한 시간을 설정함.

## 메인 게임 기능
def start_game(root, canvas, webcam_label, start_button, description_button, exit_button):
    global leftHand_wins, rightHand_wins, win_time, time_remaining  # 전역 변수를 선언합니다.

    cap = cv2.VideoCapture(0)

- 'start_game' 함수는 게임을 시작함.
- 'cap'은 웹 캠을 열어 영상을 캡처함.

## 프레임 기능 표시
    def show_frame():
        global leftHand_wins, rightHand_wins, win_time, time_remaining  # 전역 변수를 선언합니다.
        ret, img = cap.read()
        if not ret:
            return

- 'show_frame' 함수는 웹 캠에서 프레임을 읽어와 처리함.
- 영상 캡처가 실패하면 함수를 종료함.

## 손동작 인식
        if result.multi_hand_landmarks is not None:
            rps_result = []

            for res in result.multi_hand_landmarks:
                joint = np.zeros((21, 3))
                for j, lm in enumerate(res.landmark):
                    joint[j] = [lm.x, lm.y, lm.z]

                v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]
                v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]
                v = v2 - v1
                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                angle = np.arccos(np.einsum('nt,nt->n',
                                             v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                             v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))
                angle = np.degrees(angle)

                data = np.array([angle], dtype=np.float32)
                ret, results, neighbours, dist = knn.findNearest(data, 3)
                idx = int(results[0][0])

- 손 관절의 3D 좌표를 저장함.
- 관절 간 벡터를 계산하고 정규화함.
- 각도를 계산하고 KNN 모델로 제스처를 인식함.

                if idx in rps_gesture.keys():
                    org = (int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0]))
                    cv2.putText(img, text=rps_gesture[idx].upper(), org=(org[0], org[1] + 20),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

                    rps_result.append({
                        'rps': rps_gesture[idx],
                        'org': org
                    })

                mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

- 인식된 제스처를 영상에 표시함.
- 제스처 결과를 리스트에 추가함.

## 승부 결정
                if len(rps_result) >= 2:
                    winner = None
                    text = ''

                    if rps_result[0]['rps'] == 'rock':
                        if rps_result[1]['rps'] == 'rock': text = 'Tie'
                        elif rps_result[1]['rps'] == 'paper': text = 'Paper wins'; winner = 1
                        elif rps_result[1]['rps'] == 'scissors': text = 'Rock wins'; winner = 0
                    elif rps_result[0]['rps'] == 'paper':
                        if rps_result[1]['rps'] == 'rock': text = 'Paper wins'; winner = 0
                        elif rps_result[1]['rps'] == 'paper': text = 'Tie'
                        elif rps_result[1]['rps'] == 'scissors': text = 'Scissors wins'; winner = 1
                    elif rps_result[0]['rps'] == 'scissors':
                        if rps_result[1]['rps'] == 'rock': text = 'Rock wins'; winner = 1
                        elif rps_result[1]['rps'] == 'paper': text = 'Scissors wins'; winner = 0
                        elif rps_result[1]['rps'] == 'scissors': text = 'Tie'

- 두 손의 제스처를 비교하여 승자를 결정함.
- 결과에 따라 텍스트를 설정함.
