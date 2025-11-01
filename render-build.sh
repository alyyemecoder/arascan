#!/bin/bash
set -e  # Exit on error

# Install Python dependencies
pip install -r requirements.txt

# Install Node.js dependencies
cd frontend
npm install --legacy-peer-deps
npm install react-scripts@5.0.1 --save
npm run build
cd ..
