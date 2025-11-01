# Use Python 3.10 slim as the base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=10000 \
    NODE_VERSION=16.20.2 \
    NVM_DIR=/root/.nvm

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js using nvm
RUN mkdir -p $NVM_DIR \
    && curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.3/install.sh | bash \
    && . "$NVM_DIR/nvm.sh" \
    && nvm install $NODE_VERSION \
    && nvm use $NODE_VERSION \
    && nvm alias default $NODE_VERSION

# Add Node.js to PATH
ENV PATH="$NVM_DIR/versions/node/v$NODE_VERSION/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy package files
COPY requirements.txt .
COPY frontend/package*.json ./frontend/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install Node.js dependencies
RUN cd frontend && npm install --legacy-peer-deps

# Copy application code
COPY . .

# Build frontend
RUN cd frontend && npm run build

# Expose the port the app runs on
EXPOSE $PORT

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "10000"]
