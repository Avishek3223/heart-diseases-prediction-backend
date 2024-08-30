import json
import pandas as pd
import boto3
import joblib
from io import BytesIO
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)

def load_model_from_s3(bucket_name, model_key):
    s3 = boto3.client('s3')
    model_obj = s3.get_object(Bucket=bucket_name, Key=model_key)
    return joblib.load(BytesIO(model_obj['Body'].read())), model_key

def save_model_to_s3(bucket_name, model_key, model):
    s3 = boto3.client('s3')
    buffer = BytesIO()
    joblib.dump(model, buffer)
    buffer.seek(0)
    s3.put_object(Bucket=bucket_name, Key=model_key, Body=buffer)

def handler(event, context):
    start_time = time.time()

    bucket_name = event['bucket']
    majority_vote_model_key = event.get('model_keys', 'models/majority_vote_model.pkl')
    scaler_key = event['scaler_key']
    data_key = event['data']

    # Load the preprocessed heart.csv from S3
    s3 = boto3.client('s3')
    heart_obj = s3.get_object(Bucket=bucket_name, Key=data_key)
    heart_df = pd.read_csv(BytesIO(heart_obj['Body'].read()))

    # Separate features and labels
    features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    X = heart_df[features]
    y_true = heart_df['target']

    # Load scaler
    scaler_obj = s3.get_object(Bucket=bucket_name, Key=scaler_key)
    scaler = joblib.load(BytesIO(scaler_obj['Body'].read()))

    if not isinstance(scaler, (StandardScaler, MinMaxScaler)):
        return {
            'statusCode': 500,
            'body': json.dumps({'message': 'Loaded scaler is not a valid scaler object'})
        }

    # Scale the data
    X_scaled = scaler.transform(X)

    # Load the majority vote model from S3
    majority_vote_model, _ = load_model_from_s3(bucket_name, majority_vote_model_key)

    if 'votes' not in majority_vote_model:
        return {
            'statusCode': 500,
            'body': json.dumps({'message': 'Loaded majority vote model is not valid'})
        }

    votes = majority_vote_model['votes']

    # Majority vote for each instance
    majority_vote = votes.mode(axis=1)[0]

    # Calculate metrics for majority voting
    voting_accuracy = accuracy_score(y_true, majority_vote)
    voting_precision = precision_score(y_true, majority_vote)
    voting_recall = recall_score(y_true, majority_vote)
    voting_f1 = f1_score(y_true, majority_vote)

    # Metrics for majority voting
    metrics = {
        'Accuracy': voting_accuracy,
        'Precision': voting_precision,
        'Recall': voting_recall,
        'F1 Score': voting_f1
    }

    # Graph 1: Display accuracy, precision, recall, and F1 score
    plt.figure(figsize=(10, 6))
    bar_colors = ['#3498db', '#e74c3c', '#2ecc71', '#f1c40f']  # Customize colors here
    bar_width = 0.6  # Adjust bar width
    plt.bar(metrics.keys(), [v * 100 for v in metrics.values()], color=bar_colors, width=bar_width)
    
    plt.ylabel('Percentage (%)', fontsize=12)
    plt.xlabel('Metrics', fontsize=12)
    plt.title('Classification Metrics for Majority Voting', fontsize=16, fontweight='bold')
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    end_time = time.time()
    execution_time = end_time - start_time

    response = {
        'statusCode': 200,
        'body': json.dumps({
            'message': f"Metrics calculated successfully using majority vote model.",
            'metrics': metrics,
            'execution_time': execution_time,
            'majority_vote_model_key': majority_vote_model_key
        })
    }

    return response

if __name__ == "__main__":
    with open('../mocks/predict.json') as f:
        event = json.load(f)
    context = {}
    result = handler(event, context)
    print(result)
