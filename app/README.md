# Sign Language Recognition Web Application

This is a web application for real-time sign language recognition using a pre-trained model. The application uses computer vision and deep learning to recognize signs and translate them to text.

## Features

- Real-time sign language recognition through webcam
- Visual feedback with landmarks on hands, face, and pose
- Display of recognized signs as text
- User-friendly interface with instructions

## Technologies Used

- Flask: Web framework for serving the application
- TensorFlow: Deep learning framework for running the sign language recognition model
- MediaPipe: Framework for hand and pose detection
- OpenCV: Computer vision library for image processing
- HTML/CSS/JavaScript: Frontend technologies for the user interface

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Webcam or camera device

### Installation

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Make sure the model file (`action_best.h5`) is located in the parent directory.

### Running the Application

1. Start the Flask application:
   ```
   python app.py
   ```

2. Open your web browser and navigate to:
   ```
   http://127.0.0.1:5000/
   ```

## Usage

1. Position yourself in a well-lit area with the camera facing you.
2. Click the "Start Signing" button.
3. After a countdown, begin signing one of the available signs.
4. Hold each sign steady for a moment for better recognition.
5. The recognized signs will appear in the right panel.
6. Use the "Stop" button to pause recognition or "Reset" to start over.

## Available Signs

The model can recognize the following signs:
- cold
- fever
- cough
- medication
- injection
- operation
- pain

## Deployment

### Deploying to Render

For detailed deployment instructions, see the [DEPLOYMENT.md](../DEPLOYMENT.md) file in the root directory.

Quick steps:
1. Push your code to GitHub
2. Connect your repository to Render
3. Configure the web service with:
   - Build Command: `chmod +x setup.sh && ./setup.sh && pip install -r requirements.txt`
   - Start Command: `cd app && gunicorn app:app`

## License

This project is licensed under the MIT License. 