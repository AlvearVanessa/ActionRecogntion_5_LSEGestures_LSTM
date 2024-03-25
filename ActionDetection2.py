"""
Sign Detection Module for Actions
Based on: Sign Language Detection using ACTION RECOGNITION with Python | LSTM Deep Learning Model
Websites:
https://www.youtube.com/watch?v=doDUihpj6ro&t=1s
https://github.com/nicknochnack/ActionDetectionforSignLanguage

04 07 2023 10:44h CET
"""

# 6. Preprocess Data and Create Labels and Features

# Import libraries

import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp


from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

# 4. Setup Folders for Collection

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('C:/Users/maalvear/PycharmProjects/lse_vowels_gr/MP_Data')

# Actions that we try to detect
actions = np.array(['Hola', 'Gracias', 'Buenos_dias', 'Buenas_tardes', 'Buenas_noches'])

# Thirty videos worth of data
no_sequences = 30

# Videos are going to be 30 frames in length
sequence_length = 30


# Hola
## 0
## 1
## 2
## ...
## 29
# Gracias

# Buenos_dias

"""
for action in actions:
    for sequence in range(no_sequences):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

# 5. Collect Keypoint Values for Training and Testing

cap = cv2.VideoCapture(0)
# Set mediapipe model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

    # NEW LOOP
    # Loop through actions
    for action in actions:
        # Loop through sequences aka videos
        for sequence in range(no_sequences):
            # Loop through video length aka sequence length
            for frame_num in range(sequence_length):

                # Read feed
                ret, frame = cap.read()

                # Make detections
                image, results = mediapipe_detection(frame, holistic)
#                 print(results)

                # Draw landmarks
                draw_styled_landmarks(image, results)

                # NEW Apply wait logic
                if frame_num == 0:
                    cv2.putText(image, 'STARTING COLLECTION', (120,200),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(2000)
                else:
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)

                # NEW Export keypoints
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

    cap.release()
    cv2.destroyAllWindows()



# cap.release()
# cv2.destroyAllWindows()

"""


label_map = {label: num for num, label in enumerate(actions)}

print(label_map)

sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])


# Array of the features
X = np.array(sequences)
# Labels
y = to_categorical(labels).astype(int)

# Train - test split the data, test size 20% of the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)



print("X", X.shape)
print("y", y.shape)
print("X_train", X_train.shape)
print("y_train", y_train.shape)
print("X_test", X_test.shape)
print("y_test", y_test.shape)



#  7. Build and Train LSTM Neural Network

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))


# Compile the model
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
# Train the model
model.fit(X_train, y_train, epochs=200, callbacks=[tb_callback])


print(model.summary())


# 8. Make Predictions

res1 = model.predict(X_test)

print(actions[np.argmax(res1[4])])

print(actions[np.argmax(y_test[4])])

# 9. Save Weights

model.save('action200.h5')

# del model
# model.load_weights('action200.h5')




# 10. Evaluation using Confusion Matrix and Accuracy
yhat = model.predict(X_test)

ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()


print(multilabel_confusion_matrix(ytrue, yhat))

print(accuracy_score(ytrue, yhat))


