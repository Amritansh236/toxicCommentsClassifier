from pathlib import Path

# Project root directory
# __file__ is 'd:\\Toxic Comments Classifier\\src\\config.py'
# .parent is 'd:\\Toxic Comments Classifier\\src'
# .parent.parent is 'd:\\Toxic Comments Classifier'
ROOT_DIR = Path(__file__).parent.parent

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "train.csv"
SAMPLED_DATA_PATH = DATA_DIR / "train_sampled_50k.csv"

SAMPLE_SIZE = 50000

MLFLOW_TRACKING_URI = "sqlite:///" + str(ROOT_DIR / "mlflow.db")
MLFLOW_EXPERIMENT_NAME = "toxic-comment-classifier"

TARGET_COLUMNS = [
    'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'
]

