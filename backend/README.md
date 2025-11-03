# Backend API Setup

This backend processes breath analysis from accelerometer data files.

## Setup Instructions

### 1. Create a virtual environment (recommended)
```bash
python -m venv venv
```

### 2. Activate the virtual environment

**Windows:**
```bash
venv\Scripts\activate
```

**Mac/Linux:**
```bash
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the server
```bash
python app.py
```

The server will start on `http://localhost:5000`

## API Endpoints

### Health Check
```
GET http://localhost:5000/api/health
```

### Breath Analysis
```
POST http://localhost:5000/api/breath-analysis
Content-Type: multipart/form-data
Body: file (upload text file)
```

**Response:**
```json
{
  "breathing_rate": 13.32,
  "method": "envelope",
  "manual_rr": 5.08,
  "sampling_rate": 238.10,
  "dominant_freq": 0.222,
  "valid_duration": 120.45,
  "breath_count": 45,
  "status": "success"
}
```

## File Format Expected

The uploaded file should be a tab-separated text file with columns:
- Timestamp [us]
- A_X [mg]
- A_Y [mg]
- A_Z [mg]

