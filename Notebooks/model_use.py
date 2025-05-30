import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

model = load_model('emotion_detection_model.h5')

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

emoji_path = r"C:\Users\moham\Documents\GitHub\Emoji-Detection-System\emojii"  # Folder with Angry.png, Happy.png, etc.

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

print("Press ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        roi_gray = gray_frame[y:y+h, x:x+w]
        roi_resized = cv2.resize(roi_gray, (48, 48))
        roi_normalized = roi_resized.astype('float32') / 255.0
        roi_input = np.expand_dims(np.expand_dims(roi_normalized, -1), 0)

        predictions = model.predict(roi_input)
        emotion_idx = np.argmax(predictions)
        emotion_label = emotion_labels[emotion_idx]

        emoji_file = os.path.join(emoji_path, f"{emotion_label}.png")
        if os.path.exists(emoji_file):
            emoji_img = cv2.imread(emoji_file, cv2.IMREAD_UNCHANGED)
            emoji_resized = cv2.resize(emoji_img, (w, h))

            if emoji_resized.shape[2] == 4:
                alpha_s = emoji_resized[:, :, 3] / 255.0
                alpha_l = 1.0 - alpha_s

                for c in range(3):
                    frame[y:y+h, x:x+w, c] = (alpha_s * emoji_resized[:, :, c] +
                                              alpha_l * frame[y:y+h, x:x+w, c])
            else:
                frame[y:y+h, x:x+w] = emoji_resized

        label_text = f"{emotion_label} ({np.max(predictions):.2f})"
        cv2.putText(frame, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)

    cv2.imshow("Live Emotion Detection 😄", frame)

    if cv2.waitKey(10) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
