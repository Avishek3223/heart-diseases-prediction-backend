import json
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import boto3
from io import StringIO

def handler(event, context):
    # S3 bucket and file key
    bucket_name = event['bucket']
    file_key = event['key']

    print(file_key)

    # Download the dataset from S3
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=bucket_name, Key=file_key)
    data = obj['Body'].read().decode('utf-8')
    df = pd.read_csv(StringIO(data))

    # Data preprocessing
    df.fillna(df.median(), inplace=True)
    features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    X = df[features]
    y = df['target']  # Replace 'target' with the column indicating heart disease presence

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    preprocessed_data = pd.DataFrame(X_scaled, columns=features)
    preprocessed_data['target'] = y

    # Convert preprocessed data to CSV
    csv_buffer = StringIO()
    preprocessed_data.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    # Upload the preprocessed data back to S3
    s3.put_object(Bucket=bucket_name, Key='preprocessed_' + file_key, Body=csv_buffer.getvalue())

    # Convert preprocessed data to JSON
    preprocessed_json = preprocessed_data.to_json(orient='records')

    return {
        'statusCode': 200,
        'body': json.dumps({'message': 'Data preprocessed successfully', 'data': preprocessed_json})
    }

if __name__ == "__main__":
    # For local testing
    with open('test-event.json') as f:
        event = json.load(f)
    context = {}
    result = handler(event, context)
    print(result)
