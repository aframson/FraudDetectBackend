# Fraud Detection Backend

This project is the backend for the Fraud Detection application, built with FastAPI. The backend provides APIs for handling transaction data, user authentication, and fraud detection.

## Getting Started

### Prerequisites

Make sure you have the following installed on your machine:

- Python 3.8+
- MongoDB (or use MongoDB Atlas for a cloud solution)
- Pydantic

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/aframson/FraudDetect.git
   cd fraud-detection-backend
   ```
2. Create a virtual environment and activate it:

   ```bash
   python -m venv env
   source env/bin/activate  # On Windows, use `env\Scripts\activate`
   ```
3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```
4. Set up your environment variables:
   Create a `.env` file in the root of your project and add the following variables:

   ```env
   DATABASE_URL=mongodb://localhost:27017/fraud_detection
   SECRET_KEY=your_secret_key
   ```

### Running the App

Start the FastAPI server:

```bash
uvicorn app.main:app --reload
```
