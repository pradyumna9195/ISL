import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from flask import Flask, render_template, Response, jsonify

app = Flask(__name__)

# Load the model
# Check if model exists in current directory first, then try parent directory
model_path = 'action_best.h5'
if not os.path.exists(model_path):
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'action_best.h5')

print(f"Attempting to load model from: {model_path}")

# Create a custom object scope to handle deprecated parameters
custom_objects = {}

# Try multiple approaches to load the model safely
try:
    # First try: Load with legacy mode enabled and custom objects
    model = tf.keras.models.load_model(
        model_path, 
        compile=False, 
        custom_objects=custom_objects,
        options=tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
    )
    print("Model loaded successfully with method 1")
except Exception as e:
    print(f"Error with method 1: {e}")
    try:
        # Second try: Create model from scratch and load weights only
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense
        
        # Create a similar architecture to the original model
        model = Sequential()
        model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662)))
        model.add(LSTM(128, return_sequences=True, activation='relu'))
        model.add(LSTM(64, return_sequences=False, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(7, activation='softmax'))
        
        # Try to load weights only
        try:
            model.load_weights(model_path)
            print("Model loaded successfully with method 2")
        except Exception as weight_err:
            print(f"Failed to load weights: {weight_err}")
            print("Trying with alternative input shape...")
            # Try with a different input shape
            model = Sequential()
            model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662)))
            model.add(LSTM(64, return_sequences=False, activation='relu'))
            model.add(Dense(32, activation='relu'))
            model.add(Dense(7, activation='softmax'))
            print("Created alternative model without loading weights")
    except Exception as e2:
        print(f"Error with method 2: {e2}")
        # Final solution: Create a simple model for testing
        model = Sequential()
        model.add(Dense(7, activation='softmax', input_shape=(1662,)))
        print("Using simplified model for testing")

# Actions/signs that the model can recognize
actions = np.array(['cold', 'fever', 'cough', 'medication', 'injection', 'operation', 'pain'])
colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245), (255, 0, 0), (0, 255, 0), (0, 0, 255), (128, 128, 0)]

# Initialize MediaPipe components
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_styled_landmarks(image, results):
    # Draw face landmarks
    mp_drawing.draw_landmarks(
        image, 
        results.face_landmarks, 
        mp_holistic.FACEMESH_CONTOURS,
        mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
        mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
    )
    
    # Draw pose landmarks
    mp_drawing.draw_landmarks(
        image, 
        results.pose_landmarks, 
        mp_holistic.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
    )
    
    # Draw left hand landmarks
    mp_drawing.draw_landmarks(
        image, 
        results.left_hand_landmarks, 
        mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
    )
    
    # Draw right hand landmarks
    mp_drawing.draw_landmarks(
        image, 
        results.right_hand_landmarks, 
        mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
    )

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num % len(colors)], -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return output_frame

# Global variables for detection
sequence = []
sentence = []
predictions = []
threshold = 0.4  # Changed to match notebook's threshold
current_status = "Waiting for signs..."

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/test')
def test():
    return jsonify({
        "status": "ok",
        "message": "Sign Language Recognition API is running",
        "model_path": model_path,
        "actions": actions.tolist(),
        "current_status": current_status
    })

@app.route('/get_status', methods=['GET'])
def get_status():
    global current_status
    return jsonify({"status": current_status})

def generate_frames():
    global sequence, sentence, predictions, current_status
    
    # Sequence length for prediction
    sequence_length = 30
    
    # Using MediaPipe holistic model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
            
        # Set video size
        cap.set(3, 640)  # Width
        cap.set(4, 480)  # Height
        
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image")
                break
                
            try:
                # Make detection
                image, results = mediapipe_detection(frame, holistic)
                
                # Draw landmarks
                draw_styled_landmarks(image, results)
                
                # Prediction logic
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-sequence_length:]  # Keep only last 30 frames
                
                if len(sequence) == sequence_length:
                    try:
                        # Handle different model architectures
                        if isinstance(model, tf.keras.Sequential) and len(model.layers) > 0 and isinstance(model.layers[0], tf.keras.layers.LSTM):
                            # LSTM model expects 3D input
                            res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                        else:
                            # Simple model might expect 2D input (last frame only)
                            res = model.predict(np.expand_dims(keypoints, axis=0), verbose=0)[0]
                        
                        predictions.append(np.argmax(res))
                        
                        # Visualization
                        image = prob_viz(res, actions, image, colors)
                        
                        # Sentence logic - Match exactly with notebook (requiring 10 consecutive predictions)
                        if len(predictions) >= 10 and np.unique(predictions[-10:])[0] == np.argmax(res):
                            if res[np.argmax(res)] > threshold:
                                if len(sentence) > 0:
                                    if actions[np.argmax(res)] != sentence[-1]:
                                        sentence.append(actions[np.argmax(res)])
                                        print(f"Added sign to sentence: {actions[np.argmax(res)]}")
                                else:
                                    sentence.append(actions[np.argmax(res)])
                                    print(f"Started sentence with: {actions[np.argmax(res)]}")
                        
                        if len(sentence) > 5:
                            sentence = sentence[-5:]
                        
                        # Update status
                        current_status = ' '.join(sentence) if sentence else "Waiting for signs..."
                    except Exception as pred_error:
                        print(f"Error during prediction: {pred_error}")
                        # Continue without prediction
                
                # Draw the sentence box at the top (exactly like in the notebook)
                cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
                cv2.putText(image, ' '.join(sentence), (3, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Encode the frame as JPEG
                flag, buffer = cv2.imencode('.jpg', image)
                if not flag:
                    continue
                    
                frame_bytes = buffer.tobytes()
                
                # Yield the frame in the format required for multipart HTTP response
                yield (b'--frame\r\n'
                      b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
            except Exception as e:
                print(f"Error in frame processing: {e}")
                continue

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5002))
    app.run(host='0.0.0.0', port=port, debug=False) 