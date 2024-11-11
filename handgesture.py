import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress most TensorFlow logs

import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
from pynput.mouse import Button, Controller

mouse = Controller()
screen_width, screen_height = pyautogui.size()

mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1
)

def get_angle(a, b, c):
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(np.degrees(radians))
    return angle

def get_distance(landmark_list):
    if len(landmark_list) < 2:
        return
    (x1, y1), (x2, y2) = landmark_list[0], landmark_list[1]
    L = np.hypot(x2 - x1, y2 - y1)
    return np.interp(L, [0, 1], [0, 1000])

def find_finger_tip(processed):
    if processed.multi_hand_landmarks:
        hand_landmarks = processed.multi_hand_landmarks[0]  # Assuming only one hand is detected
        index_finger_tip = hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP]
        return index_finger_tip
    return None

def move_mouse(index_finger_tip):
    if index_finger_tip is not None:
        x = int(index_finger_tip.x * screen_width)
        y = int(index_finger_tip.y * screen_height)
        pyautogui.moveTo(x, y)

def is_left_click(landmark_list, thumb_index_dist):
    return (
        get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 50 and
        get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) > 90 and
        thumb_index_dist > 50
    )

def is_right_click(landmark_list, thumb_index_dist):
    # Increase angle threshold to make right-click gesture more distinct
    return (
        get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 45 and
        get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) > 95 and
        thumb_index_dist > 70  # Increase distance threshold
    )


def is_double_click(landmark_list, thumb_index_dist):
    return (
        get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 50 and
        get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50 and
        thumb_index_dist > 50
    )

def is_scroll(landmark_list, thumb_index_dist):
    return (
        get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 50 and
        get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50 and
        thumb_index_dist < 50
    )

def detect_gesture(frame, landmark_list, processed):
    if len(landmark_list) >= 21:
        index_finger_tip = find_finger_tip(processed)
        thumb_index_dist = get_distance([landmark_list[4], landmark_list[5]])

        if get_distance([landmark_list[4], landmark_list[5]]) < 50 and get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) > 90:
            move_mouse(index_finger_tip)
        elif is_left_click(landmark_list, thumb_index_dist):
            mouse.press(Button.left)
            mouse.release(Button.left)
            cv2.putText(frame, "Left Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif is_right_click(landmark_list, thumb_index_dist):
            mouse.press(Button.right)
            mouse.release(Button.right)
            cv2.putText(frame, "Right Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif is_double_click(landmark_list, thumb_index_dist):
            pyautogui.doubleClick()
            cv2.putText(frame, "Double Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        elif is_scroll(landmark_list, thumb_index_dist):
            # Debugging: Print thumb_index_dist to understand the values for scrolling
            print(f"Thumb index distance: {thumb_index_dist}")
            if thumb_index_dist > 0:  # Assuming thumb_index_dist represents the y-distance
                pyautogui.scroll(50)  # Scroll up
                cv2.putText(frame, "Scroll Up", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            elif thumb_index_dist < 0:
                pyautogui.scroll(-50)  # Scroll down
                cv2.putText(frame, "Scroll Down", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

def count_fingers(lst):
    cnt = 0
    thresh = (lst.landmark[0].y - lst.landmark[9].y) * 0.5

    if (lst.landmark[5].y - lst.landmark[8].y) > thresh:  # Index finger
        cnt += 1
    if (lst.landmark[9].y - lst.landmark[12].y) > thresh:  # Middle finger
        cnt += 1
    if (lst.landmark[13].y - lst.landmark[16].y) > thresh:  # Ring finger
        cnt += 1
    if (lst.landmark[17].y - lst.landmark[20].y) > thresh:  # Pinky
        cnt += 1
    if (lst.landmark[4].x < lst.landmark[3].x and (lst.landmark[5].x - lst.landmark[4].x) > 0.1):  # Adjusted thumb detection
        cnt += 1

    return cnt


def control_video(cnt):
    if cnt == 1:
        pyautogui.press("right")
        time.sleep(0.3)  # Add a short delay
    elif cnt == 2:
        pyautogui.press("left")
        time.sleep(0.3)  # Add a short delay
    elif cnt == 3:
        pyautogui.press("up")
        time.sleep(0.3)  # Add a short delay
    elif cnt == 4:
        pyautogui.press("down")
        time.sleep(0.3)  # Add a short delay
    elif cnt == 5:
        pyautogui.press("space")
        time.sleep(0.3)  # Add a short delay


def main():
    draw = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)

    start_init = False
    prev = -1

    cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Frame', screen_width, screen_height)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed = hands.process(frameRGB)

            landmark_list = []
            if processed.multi_hand_landmarks:
                hand_landmarks = processed.multi_hand_landmarks[0]  # Assuming only one hand is detected
                draw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)
                for lm in hand_landmarks.landmark:
                    landmark_list.append((lm.x, lm.y))

                detect_gesture(frame, landmark_list, processed)

                # Video control gestures
                cnt = count_fingers(hand_landmarks)
                if prev != cnt:
                    if not start_init:
                        start_time = time.time()
                        start_init = True
                    elif (time.time() - start_time) > 0.2:  # Correct timing
                        control_video(cnt)
                        prev = cnt
                        start_init = False

            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
