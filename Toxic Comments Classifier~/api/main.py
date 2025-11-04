import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import nltk
from pathlib import Path
import os
import sys

APP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(APP_ROOT)

from src.preprocess import clean_text, download_nltk_data
    
# --- Pydantic Models ---

class Comment(BaseModel):
    text: str

class Prediction(BaseModel):
    label: str
    is_toxic: bool

# --- FastAPI App ---

app = FastAPI(
    title="Toxic Comment Detection API",
    description="An API to classify social media comments as toxic or non-toxic.",
    version="1.0.0"
)

# --- Model Loading ---

model = None

@app.on_event("startup")
async def load_model_and_nltk():
    """
    On startup, load the model from the local MLflow artifacts
    and download NLTK data.
    """
    global model
    
    print("Downloading NLTK data for API...")
    download_nltk_data()
    print("NLTK data ready.")

    APP_ROOT = Path(__file__).resolve().parent.parent 

    model_path = (
        APP_ROOT / 
        "mlruns" / 
        "1" / 
        "models" / 
        "m-6629f18f7b4645f8bdcb561378e8a4bc" / 
        "artifacts"
    )
    

    model_uri = model_path.as_uri()

    print(f"Loading model from local URI: {model_uri}")
    
    try:
        model = mlflow.sklearn.load_model(model_uri)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Fatal error loading model from path: {model_uri}")
        print(f"Error: {e}")
        print("---")
        print(f"Check 1: Does this folder exist? {model_path}")
        print("---")
        raise e  

@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Toxic Comment Classifier API. Post to /predict"
    }

@app.post("/predict", response_model=Prediction)
async def predict(comment: Comment):
    """
    Predicts if a given comment is toxic or non-toxic.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Check server logs. Was MLFLOW_RUN_ID set?"
        )

    try:
        cleaned_text = clean_text(comment.text)
        
        prediction_val = model.predict([cleaned_text])[0]
        
        is_toxic = bool(prediction_val)
        label = "toxic" if is_toxic else "non-toxic"
        
        return Prediction(label=label, is_toxic=is_toxic)
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {e}"
        )

if __name__ == "__main__":
    import uvicorn
    print("--- Starting local test server ---")
    print("WARNING: MLFLOW_RUN_ID must be set in your environment for this to work.")
    print("Example: export MLFLOW_RUN_ID=your_run_id")
    print("Example: export MLFLOW_TRACKING_URI=sqlite:///../mlflow.db")
    
    os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///../mlflow.db")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
