Toxic Comment Detection API (MLOps Project)

This is a complete, end-to-end MLOps project that builds, tracks, and deploys a lightweight NLP system for classifying social media comments as "toxic" or "non-toxic."

The project follows a full MLOps workflow:

Data Ingestion & Preprocessing: Cleaning and sampling the Jigsaw dataset.

Model Training: Training a TF-IDF + Logistic Regression pipeline.

Experiment Tracking: Using MLflow to log parameters, metrics (like F1-score and Recall), and model artifacts.

Model Deployment: Containerizing the prediction API with Docker.

Real-Time Serving: Serving the model via a FastAPI endpoint.

Interactive UI: A Streamlit dashboard to interact with the live API.

Features

Real-Time API: A FastAPI endpoint (/predict) that takes new text and returns a JSON classification.

Experiment Tracking: All training runs are logged in MLflow, allowing for easy comparison of different models (e.g., a biased vs. a balanced model).

Containerized & Reproducible: The entire prediction service is packaged in a Docker container, so it can run anywhere.

Interactive Dashboard: A simple Streamlit app to visualize training data and test the live API.

Data-Driven Iteration: The train.py script is designed to be run multiple times, and the API can be updated to serve a new model simply by changing a RUN_ID in the docker run command.

Tech Stack

Python 3.10

Model: scikit-learn (TF-IDF + LogisticRegression)

Experiment Tracking: MLflow

API Server: FastAPI & Uvicorn

Containerization: Docker

Dashboard: Streamlit

Data Handling: Pandas & NLTK

Project Structure

toxic-comment-api/
├── api/
│   ├── main.py           # FastAPI app (loads model from mlruns)
│   ├── Dockerfile        # Docker recipe for the API
│   └── requirements.txt  # Python reqs for API
├── dashboard/
│   └── app.py            # Streamlit dashboard UI
├── data/
│   └── .gitkeep          # Data folder (see setup instructions)
├── src/
│   ├── config.py         # Project configuration (paths, params)
│   ├── preprocess.py     # Text cleaning function
│   └── train.py          # Model training & MLflow logging
├── mlruns/
│   └── .gitkeep          # MLflow artifacts (created by train.py)
├── .dockerignore
├── .gitignore
├── README.md             
└── requirements.txt      # Main Python reqs (training, streamlit)


🚀 How to Run This Project

Prerequisites

Python 3.10+

Git

Docker Desktop (must be running in the background)

Step 1: Clone the Repository

git clone [https://github.com/YourUsername/toxic-comment-api.git](https://github.com/YourUsername/toxic-comment-api.git)
cd toxic-comment-api


Step 2: Set Up the Data (Important!)

Note: The raw training data (train.csv) is several gigabytes and is not included in this repository (it is listed in .gitignore).

You must download it manually from Kaggle.

Go to the Kaggle: Jigsaw Toxic Comment Classification Challenge.

Download the train.csv.zip file.

Unzip it and place train.csv inside the data/ folder in this project.

The final path should be: toxic-comment-api/data/train.csv.

Step 3: Install Main Dependencies

This installs the packages needed for training and the dashboard.

pip install -r requirements.txt


Step 4: Train Your First Model

This script will load data/train.csv, sample it, clean it, train a model, and (most importantly) save all the model files into the mlruns folder.

python src/train.py


When it finishes, it will print a Run ID. Copy this ID!

--- MLflow Run Summary ---
Experiment: toxic-comment-classifier
Run ID: 69132c4bfc2745acb4c683ee7dfdcf2b  <-- COPY THIS ID!
...
--------------------------


(You can also view all your runs by running mlflow ui in a separate terminal and opening http://localhost:5000)

Step 5: Build and Run the Docker API

Now, we will build the Docker image and run it, telling it to load the model you just trained.

Build the image:
(This only needs to be done once)

docker build -t toxic-api -f api/Dockerfile .


Run the container:
(Replace <YOUR_RUN_ID_HERE> with the Run ID you copied)

docker run -e MLFLOW_RUN_ID="<YOUR_RUN_ID_HERE>" -p 8000:8000 toxic-api


You will see logs from Uvicorn, ending in:
INFO: Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)

Your API is now live!

Step 6: View the Streamlit Dashboard

Open a new terminal (leave the Docker API running!).

Run the Streamlit app:

streamlit run dashboard/app.py


Your browser will automatically open to http://localhost:8501, where you can see the data visualizations and test your live API.

Project Insights: The Class Imbalance Problem

A key finding from this project was the severe class imbalance in the training data, where over 90% of comments were "non-toxic."

This created an "Accuracy Trap": a lazy model that always predicted "non-toxic" would be 90% accurate but completely useless. This was visible in the initial model's metrics, which had high accuracy but a very low Recall (it was missing most of the toxic comments).

The Solution:
The train.py script was updated to use class_weight='balanced' in the LogisticRegression model. This "punishes" the model more for misclassifying the rare toxic class.

As seen in the MLflow comparison, this created a Precision-Recall Trade-off:

V1 Model (Biased): High Accuracy, High Precision, Very Low Recall.

V2 Model (Balanced): Lower Accuracy, Lower Precision, Much Higher Recall.

The V2 model is far more useful for content moderation, as its primary goal is to find (or "recall") toxic comments, even if it means flagging a few innocent ones (like "i love you," which it learned to associate with sarcasm).
