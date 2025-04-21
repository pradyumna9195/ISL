# Deploying to Render

This guide explains how to deploy the Sign Language Recognition application to Render.

## Prerequisites

- A [Render](https://render.com/) account
- Your project code pushed to a GitHub repository

## Deployment Steps

### 1. Push Your Code to GitHub

Make sure your repository includes:
- The application code in the `app/` directory
- The model file `action_best.h5` in the root directory
- The `render.yaml` configuration file
- The `requirements.txt` file
- The `setup.sh` script

### 2. Deploy to Render

#### Using the Deploy Button (Easiest)

If you've set up the `render.yaml` file correctly, you can create a "Deploy to Render" button in your GitHub repository. Add the following to your README.md:

```markdown
[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)
```

#### Manual Deployment

1. Log in to your Render account
2. Go to the Dashboard and click "New +"
3. Select "Web Service"
4. Connect your GitHub repository
5. Configure the service:
   - Name: `isl-sign-language-recognition` (or your preferred name)
   - Environment: `Python`
   - Build Command: `chmod +x setup.sh && ./setup.sh && pip install -r requirements.txt`
   - Start Command: `cd app && gunicorn app:app`
   - Select the appropriate instance type (Free tier is fine for testing)
6. Click "Create Web Service"

### 3. Environment Variables

If needed, add the following environment variables in the Render dashboard:
- `PYTHON_VERSION`: `3.10.0`

### 4. Verify Deployment

Once deployment is complete, Render will provide a URL for your application. Visit this URL to ensure your application is running correctly.

## Troubleshooting

### Webcam Issues

The application uses webcam access, which might not work in a cloud environment. For a production application, you should consider:

1. Creating a mobile app that captures video and sends frames to the backend
2. Providing file upload functionality for video processing
3. Implementing WebRTC for real-time video

### Model Loading Issues

If the model doesn't load correctly, check the Render logs for errors. You may need to:

1. Ensure the model file was correctly copied to the app directory
2. Verify the model format is compatible with the TensorFlow version
3. Check if the model path is correct in the application code 