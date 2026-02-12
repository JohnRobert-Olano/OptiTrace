
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
import time
import os

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_PATH = os.path.join("models", "best_model.keras")
TASK_PATH = "face_landmarker.task"
IMG_SIZE = (224, 224)
CLASS_NAMES = ["cataract", "conjunctivitis", "normal", "uveitis"]

# Landmark Indices for Eyes (Refined Landmarks - same as legacy but accessed via result)
LEFT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    print(f"Loading model from {MODEL_PATH}...")
    try:
        model = load_model(MODEL_PATH)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def get_landmark_points(landmarks, indices, width, height):
    points = []
    for idx in indices:
        lm = landmarks[idx]
        x, y = int(lm.x * width), int(lm.y * height)
        points.append([x, y])
    return np.array(points, dtype=np.int32)

def clipper_preprocess(frame, landmarks):
    h, w, _ = frame.shape
    
    # 1. Get Eye Points
    # Landmarks is a list of objects with x, y, z attributes
    left_eye_pts = get_landmark_points(landmarks, LEFT_EYE_INDICES, w, h)
    right_eye_pts = get_landmark_points(landmarks, RIGHT_EYE_INDICES, w, h)
    
    # 2. Create Mask
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, left_eye_pts, 255)
    cv2.fillConvexPoly(mask, right_eye_pts, 255)
    
    # 3. Apply Mask
    mask_3ch = cv2.merge([mask, mask, mask])
    masked_frame = cv2.bitwise_and(frame, mask_3ch)
    
    # 4. Crop to Bounding Box
    all_points = np.vstack((left_eye_pts, right_eye_pts))
    x, y, w_box, h_box = cv2.boundingRect(all_points)
    
    pad = 20
    x = max(0, x - pad)
    y = max(0, y - pad)
    w_box = min(w - x, w_box + 2 * pad)
    h_box = min(h - y, h_box + 2 * pad)
    
    cropped_eye_region = masked_frame[y:y+h_box, x:x+w_box]
    
    if cropped_eye_region.size == 0:
        return None, None
        
    # 5. Resize
    input_tensor = cv2.resize(cropped_eye_region, IMG_SIZE)
    input_tensor = preprocess_input(input_tensor.astype(np.float32))
    input_tensor = np.expand_dims(input_tensor, axis=0) 
    
    return input_tensor, (x, y, w_box, h_box)

def main():
    if not os.path.exists(TASK_PATH):
        print(f"Error: {TASK_PATH} not found. Please download it.")
        return

    model = load_trained_model()
    if model is None:
        return

    # MediaPipe Task Setup
    base_options = python.BaseOptions(model_asset_path=TASK_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting Camera... Press 'q' to quit.")
    
    fps_start_time = time.time()
    fps_frame_count = 0
    fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to MP Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Detect
        detection_result = detector.detect(mp_image)
        
        prediction_text = "Searching for eyes..."
        confidence_text = ""
        color = (0, 255, 255)
        
        if detection_result.face_landmarks:
            # We only requested 1 face
            face_landmarks = detection_result.face_landmarks[0]
            
            # Preprocess
            input_tensor, bbox = clipper_preprocess(frame, face_landmarks)
            
            if input_tensor is not None:
                preds = model.predict(input_tensor, verbose=0)
                idx = np.argmax(preds)
                confidence = preds[0][idx] * 100
                disease = CLASS_NAMES[idx]
                
                if disease == "normal":
                    color = (0, 255, 0)
                else:
                    color = (0, 0, 255)
                    
                prediction_text = f"Result: {disease.upper()}"
                confidence_text = f"Conf: {confidence:.1f}%"
                
                x, y, w_box, h_box = bbox
                cv2.rectangle(frame, (x, y), (x+w_box, y+h_box), color, 2)
                
                # Draw contours (visual debug)
                h_img, w_img, _ = frame.shape
                left_pts = get_landmark_points(face_landmarks, LEFT_EYE_INDICES, w_img, h_img)
                right_pts = get_landmark_points(face_landmarks, RIGHT_EYE_INDICES, w_img, h_img)
                cv2.polylines(frame, [left_pts], True, (255, 255, 0), 1)
                cv2.polylines(frame, [right_pts], True, (255, 255, 0), 1)

        fps_frame_count += 1
        if time.time() - fps_start_time >= 1.0:
            fps = fps_frame_count / (time.time() - fps_start_time)
            fps_frame_count = 0
            fps_start_time = time.time()
            
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 60), (0, 0, 0), -1)
        cv2.putText(frame, prediction_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        if confidence_text:
            cv2.putText(frame, confidence_text, (350, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 120, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)

        cv2.imshow('OptiTrace Real-time Tester', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
