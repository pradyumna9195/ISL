#!/bin/bash

# Copy the model file to the app directory if it exists at the root level
if [ -f "action_best.h5" ]; then
    echo "Copying model file to app directory..."
    cp action_best.h5 app/
fi

echo "Setup complete." 