# Sign Language Recognition System

This project implements a sign language recognition system that can identify three different signs: "hello", "thanks", and "I love you". The system uses computer vision and deep learning techniques to recognize hand gestures in real-time.

## Project Structure

- `isl1.ipynb`: Jupyter notebook containing the model training code
- `action.h5`: Trained model file
- `app/`: Web application for real-time sign language recognition
  - `app.py`: Flask application
  - `templates/`: HTML templates
  - `static/`: CSS and JavaScript files
  - `requirements.txt`: Dependencies

## Web Application

The web application provides a user-friendly interface for real-time sign language recognition. It uses the webcam to capture video, processes the frames to detect hand gestures, and displays the recognized signs as text.

### Features

- Real-time sign language recognition
- Visual feedback with landmarks on hands, face, and pose
- Display of recognized signs as text
- User-friendly interface with instructions

## Model Training

The model was trained using the following steps:

1. Data collection: Recording videos of different sign language gestures
2. Feature extraction: Using MediaPipe to extract keypoints from hands, face, and pose
3. Model training: Using a LSTM neural network to recognize patterns in the keypoint sequences
4. Model evaluation: Testing the model on unseen data

## Getting Started

To run the web application, follow these steps:

1. Navigate to the app directory:
   ```
   cd app
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Start the Flask application:
   ```
   python app.py
   ```

4. Open your web browser and navigate to:
   ```
   http://127.0.0.1:5000/
   ```

For more detailed instructions, see the README.md file in the app directory.

## License

This project is licensed under the MIT License. 