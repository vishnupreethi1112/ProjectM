# import os
# import pickle

# import mediapipe as mp
# import cv2
# import matplotlib.pyplot as plt


# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles

# hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# DATA_DIR = './data'

# data = []
# labels = []
# for dir_ in os.listdir(DATA_DIR):
#     for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
#         data_aux = []

#         x_ = []
#         y_ = []

#         img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#         results = hands.process(img_rgb)
#         if results.multi_hand_landmarks:
#             for hand_landmarks in results.multi_hand_landmarks:
#                 for i in range(len(hand_landmarks.landmark)):
#                     x = hand_landmarks.landmark[i].x
#                     y = hand_landmarks.landmark[i].y

#                     x_.append(x)
#                     y_.append(y)

#                 for i in range(len(hand_landmarks.landmark)):
#                     x = hand_landmarks.landmark[i].x
#                     y = hand_landmarks.landmark[i].y
#                     data_aux.append(x - min(x_))
#                     data_aux.append(y - min(y_))

#             data.append(data_aux)
#             labels.append(dir_)

# f = open('data.pickle', 'wb')
# pickle.dump({'data': data, 'labels': labels}, f)
# f.close()

import os
import pickle
import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Use static_image_mode for processing images in a dataset
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

data = []
labels = []
for dir_ in os.listdir(DATA_DIR):
    # Convert the label from string to int
    label = int(dir_)  # Assuming directory names are numeric (0-26 for A-Z and SPACE)

    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []
        x_ = []
        y_ = []

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        if img is None:
            print(f"Error: Unable to read image {os.path.join(DATA_DIR, dir_, img_path)}. Skipping.")
            continue  # Skip if the image cannot be read

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                # Normalize the landmark coordinates
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            data.append(data_aux)
            labels.append(label)  # Use the integer label for the class

f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()

print("Data processing complete. Data saved to 'data.pickle'.")
