#!/bin/bash

# Echo all commands and stop on errors
set -ex

# Create app directory if it doesn't exist
if [ ! -d "app" ]; then
    echo "Creating app directory..."
    mkdir -p app
fi

# Copy the model file to the app directory if it exists at the root level
if [ -f "action_best.h5" ]; then
    echo "Copying model file to app directory..."
    cp action_best.h5 app/
    echo "Model file copied successfully."
else
    echo "WARNING: Model file not found in root directory!"
    # Check if it exists in other common locations
    if [ -f "../action_best.h5" ]; then
        echo "Found model file in parent directory, copying..."
        cp ../action_best.h5 app/
    fi
fi

# List files in app directory to verify
echo "Files in app directory:"
ls -la app/

echo "Setup complete." 