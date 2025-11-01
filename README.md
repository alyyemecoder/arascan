# Arabic Character Recognition

A FastAPI backend with React frontend for Arabic character recognition using YOLOv8.

## Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+
- npm or yarn

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd arascan
   ```

2. **Set up Python environment**
   ```bash
   # Create and activate virtual environment
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Linux/Mac:
   # source venv/bin/activate

   # Install Python dependencies
   pip install -r requirements.txt
   ```

3. **Set up Frontend**
   ```bash
   cd frontend
   npm install
   npm run build
   cd ..
   ```

4. **Environment Variables**
   Create a `.env` file in the root directory with:
   ```
   # Server Configuration
   HOST=0.0.0.0
   PORT=8000
   
   # CORS Settings
   CORS_ORIGINS=["http://localhost:3000"]
   
   # Uploads
   UPLOAD_DIR=./uploads
   DEBUG=True
   ```

### Running the Application

1. **Start the backend server**
   ```bash
   uvicorn app.main:app --reload
   ```

2. **Start the frontend development server** (in a new terminal)
   ```bash
   cd frontend
   npm start
   ```

3. Open your browser to `http://localhost:3000`

## Project Structure

```
arascan/
├── app/                  # Backend application
├── frontend/             # React frontend
├── models/               # ML models
├── uploads/              # Uploaded files
├── .gitignore           
├── requirements.txt     
└── README.md
```

## Deployment

### Production Build
```bash
# Build frontend
cd frontend
npm run build

# Run production server
cd ..
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License.
