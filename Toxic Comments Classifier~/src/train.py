import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, f1_score
import mlflow
import mlflow.sklearn
import joblib
import os
import numpy as np
from mlflow.models import infer_signature

import config
from preprocess import clean_text, download_nltk_data

def load_and_sample_data():
    """
    Loads raw data, samples it, and saves the sample if it doesn't exist.
    This speeds up subsequent runs.
    """
    if os.path.exists(config.SAMPLED_DATA_PATH):
        print(f"Loading sampled data from {config.SAMPLED_DATA_PATH}...")
        return pd.read_csv(config.SAMPLED_DATA_PATH)
        
    print(f"Loading raw data from {config.RAW_DATA_PATH}...")
    try:
        df = pd.read_csv(config.RAW_DATA_PATH)
    except FileNotFoundError:
        print(f"Error: Raw data file not found at {config.RAW_DATA_PATH}")
        print("Please download it from Kaggle and place it in the data/ directory.")
        return None

    print(f"Sampling {config.SAMPLE_SIZE} rows...")
    df_sampled = df.sample(n=config.SAMPLE_SIZE, random_state=42)
    
    config.DATA_DIR.mkdir(exist_ok=True)
    
    print(f"Saving sampled data to {config.SAMPLED_DATA_PATH}...")
    df_sampled.to_csv(config.SAMPLED_DATA_PATH, index=False)
    
    return df_sampled

def main():
    print("Checking NLTK data...")
    download_nltk_data()

    df = load_and_sample_data()
    if df is None:
        return

    print("Preprocessing text data...")
    df['comment_text'] = df['comment_text'].fillna('') 
    df['cleaned_comment'] = df['comment_text'].apply(clean_text)

    df['is_toxic'] = (df[config.TARGET_COLUMNS].sum(axis=1) > 0).astype(int)

    X = df['cleaned_comment']
    y = df['is_toxic']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)

    print("Starting MLflow run with BALANCED CLASS WEIGHTS...")
    with mlflow.start_run() as run:
    
        tfidf_max_features = 5000
        lr_C = 1.0
        lr_solver = 'liblinear'
        class_weight = 'balanced'

        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=tfidf_max_features)),
            ('clf', LogisticRegression(
                C=lr_C, 
                solver=lr_solver, 
                class_weight=class_weight, # <--- THE FIX
                random_state=42
            ))
        ])

        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("vectorizer", "TfidfVectorizer")
        mlflow.log_param("tfidf_max_features", tfidf_max_features)
        mlflow.log_param("lr_C", lr_C)
        mlflow.log_param("lr_solver", lr_solver)
        mlflow.log_param("sample_size", config.SAMPLE_SIZE)
        mlflow.log_param("class_weight", class_weight) # <--- LOG THE FIX

        print("Training model...")
        pipeline.fit(X_train, y_train)
        print("Training complete.")

        y_pred = pipeline.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        report = classification_report(y_test, y_pred, output_dict=True)
        
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_macro", f1_macro)
        mlflow.log_metric("precision_toxic", report['1']['precision'])
        mlflow.log_metric("recall_toxic", report['1']['recall'])
        mlflow.log_metric("f1_toxic", report['1']['f1-score'])

        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 (Macro): {f1_macro:.4f}")

        print("Logging model to MLflow...")
        signature = infer_signature(X_test, y_pred)
        input_example = {"text": "This is an example comment."}
        

        input_example_df = pd.DataFrame({"text": ["this is an example", "this is another"]})
        signature = infer_signature(input_example_df, pipeline.predict(input_example_df["text"]))

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            signature=signature,
            input_example=input_example_df,
            registered_model_name="toxic-comment-pipeline"
        )

        print("\n--- MLflow Run Summary ---")
        print(f"Experiment: {config.MLFLOW_EXPERIMENT_NAME}")
        print(f"Run ID: {run.info.run_id}")
        print("Model, parameters, and metrics logged successfully.")
        print(f"Run 'mlflow ui --backend-store-uri {config.MLFLOW_TRACKING_URI}' to view.")
        print("--------------------------\n")

if __name__ == "__main__":
    main()

