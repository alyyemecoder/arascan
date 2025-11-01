#!/bin/bash

# Create and activate Python virtual environment
echo "Setting up Python environment..."
python -m venv venv
source venv/bin/activate  # On Windows, use: .\venv\Scripts\activate

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Set up frontend
echo "Setting up frontend..."
cd frontend
npm install
npm run build
cd ..

echo "Setup complete!"
echo "To activate the virtual environment, run:"
echo "On Linux/Mac: source venv/bin/activate"
echo "On Windows: .\venv\Scripts\activate"
