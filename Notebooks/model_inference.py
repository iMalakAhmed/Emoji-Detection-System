import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("emotion_detection_model.h5")

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
label_to_emoji = {
    'Angry': '😠',
    'Disgust': '🤢',
    'Fear': '😨',
    'Happy': '😄',
    'Sad': '😢',
    'Surprise': '😲',
    'Neutral': '😐'
}

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    img_resized = cv2.resize(img, (48, 48))
    img_normalized = img_resized.astype("float32") / 255.0
    img_input = img_normalized.reshape(1, 48, 48, 1)
    return img_input, img_resized

def predict_emotion_with_emoji(image_path):
    input_tensor, img_display = preprocess_image(image_path)
    predictions = model.predict(input_tensor)
    emotion_idx = np.argmax(predictions)
    label = emotion_labels[emotion_idx]
    emoji = label_to_emoji[label]
    confidence = predictions[0][emotion_idx]

    print(f" Predicted Emotion: {label}")
    print(f" Emoji Output: {emoji}")
    print(f"Confidence: {confidence:.2f}")

    display_img = cv2.cvtColor(img_display, cv2.COLOR_GRAY2BGR)
    cv2.putText(display_img, f"{emoji} {label}", (2, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imshow("Predicted Emotion", display_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = r"C:\Users\moham\Documents\GitHub\Emoji-Detection-System\FER-2013\train\surprise\Training_163503.jpg"  # ← replace with your own image
    predict_emotion_with_emoji(image_path)
