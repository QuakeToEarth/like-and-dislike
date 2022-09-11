from turtle import width
import cv2
import mediapipe as mp
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_join = mp.solutions.drawing_utils
camera = cv2.VideoCapture(0)
fingerTip = [8, 12, 16, 20]
thumbTip = 4
while True:
    ret, img = camera.read()
    img = cv2.flip(img, 1)
    height, width, channels = img.shape
    results = hands.process(img)
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks
        for land in hand_landmarks:
            lmList = []
            for id, lm in enumerate(land.landmark):
                lmList.append(lm)
            fingerFoldStatus = []
            for eachTip in fingerTip:
                if lmList[eachTip].x > lmList[eachTip-3].x:
                    fingerFoldStatus.append(True)
                else:
                    fingerFoldStatus.append(False)
            if all(fingerFoldStatus) == True:
                if lmList[thumbTip].y < lmList[thumbTip-1].y < lmList[thumbTip-2].y:
                    cv2.putText(img, "LIKE", (20, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                if lmList[thumbTip].y > lmList[thumbTip-1].y > lmList[thumbTip-2].y:
                    cv2.putText(img, "DISLIKE", (20, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            mp_join.draw_landmarks(img, land, mp_hands.HAND_CONNECTIONS, mp_join.DrawingSpec(
                (0, 0, 255), 2, 2), mp_join.DrawingSpec((0, 255, 0), 4, 2))
            cv2.imshow("Hand Tracking...", img)
            cv2.waitKey(1)
