import json
import pandas as pd
import boto3
import joblib
from io import BytesIO
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

def load_model_from_s3(bucket_name, model_key):
    s3 = boto3.client('s3')
    model_obj = s3.get_object(Bucket=bucket_name, Key=model_key)
    return joblib.load(BytesIO(model_obj['Body'].read())), model_key

def handler(event, context):
    start_time = time.time()

    bucket_name = event['bucket']
    model_keys = event['model_keys']
    scaler_key = event['scaler_key']

    user_data = event['data']
    features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

    normal_values = {
        'age': 55,
        'sex': 1,
        'cp': 0,
        'trestbps': 120,
        'chol': 200,
        'fbs': 0,
        'restecg': 1,
        'thalach': 150,
        'exang': 0,
        'oldpeak': 0.0,
        'slope': 2,
        'ca': 0,
        'thal': 2
    }

    complete_data = {feature: user_data.get(feature, normal_values[feature]) for feature in features}
    user_df = pd.DataFrame([complete_data], columns=features)

    # Load scaler
    s3 = boto3.client('s3')
    scaler_obj = s3.get_object(Bucket=bucket_name, Key=scaler_key)
    scaler = joblib.load(BytesIO(scaler_obj['Body'].read()))

    print("Loaded scaler type:", type(scaler))

    if not isinstance(scaler, (StandardScaler, MinMaxScaler)):
        return {
            'statusCode': 500,
            'body': json.dumps({'message': 'Loaded scaler is not a valid scaler object'})
        }

    try:
        user_data_scaled = scaler.transform(user_df)
    except ValueError as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'message': f'Scaler transformation error: {str(e)}'})
        }

    votes = []
    model_predictions = {}

    # Parallel loading of models and predictions
    with ThreadPoolExecutor(max_workers=len(model_keys)) as executor:
        future_to_model = {executor.submit(load_model_from_s3, bucket_name, model_key): model_key for model_key in model_keys}
        for future in as_completed(future_to_model):
            model_key = future_to_model[future]
            try:
                model, model_key = future.result()
                prediction = model.predict(user_data_scaled)
                votes.append(int(prediction[0]))
                model_predictions[model_key] = int(prediction[0])
                print(f"Model: {model_key}, Prediction: {int(prediction[0])}")
            except ModuleNotFoundError as e:
                error_message = f"Error loading model {model_key}: {str(e)}"
                print(error_message)
                return {
                    'statusCode': 500,
                    'body': json.dumps({'message': error_message})
                }
            except Exception as e:
                error_message = f"Unexpected error loading model {model_key}: {str(e)}"
                print(error_message)
                return {
                    'statusCode': 500,
                    'body': json.dumps({'message': error_message})
                }

    majority_vote = 1 if votes.count(1) > votes.count(0) else 0
    result = 'Heart Disease Detected' if majority_vote == 1 else 'No Heart Disease Detected'

    print("Votes:", votes)
    print("Majority Vote Result:", result)

    missing_features = [feature for feature in features if feature not in user_data]

    response_message = f"Based on provided data, the result is: {result}. "
    if missing_features:
        response_message += (
            "These values were taken as normal: "
            f"{', '.join(missing_features)}. "
            "Please provide this data for a more accurate result."
        )

    end_time = time.time()
    execution_time = end_time - start_time

    response = {
        'statusCode': 200,
        'body': json.dumps({
            'message': response_message,
            'result': result,
            'model_predictions': model_predictions,
            'execution_time': execution_time
        })
    }

    return response

if __name__ == "__main__":
    # Adjusted path to access 'predict.json' in the 'mocks' directory
    with open('../mocks/predict.json') as f:
        event = json.load(f)
    context = {}
    result = handler(event, context)
    print(result)
