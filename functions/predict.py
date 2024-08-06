import json
import pandas as pd
import boto3
import joblib  # Use joblib for loading scalers
from io import BytesIO  # Import BytesIO for in-memory binary streams
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def handler(event, context):
    bucket_name = event['bucket']
    model_keys = event['model_keys']
    scaler_key = event['scaler_key']

    user_data = event['data']
    features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    
    # Define normal values for each feature
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
    
    # Fill missing features with normal values
    complete_data = {feature: user_data.get(feature, normal_values[feature]) for feature in features}

    user_df = pd.DataFrame([complete_data], columns=features)

    s3 = boto3.client('s3')
    scaler_obj = s3.get_object(Bucket=bucket_name, Key=scaler_key)
    scaler = joblib.load(BytesIO(scaler_obj['Body'].read()))  # Use joblib for loading

    # Debug information
    print("Loaded scaler type:", type(scaler))

    if not isinstance(scaler, (StandardScaler, MinMaxScaler)):
        return {
            'statusCode': 500,
            'body': json.dumps({'message': 'Loaded scaler is not a valid scaler object'})
        }

    # Scale the input data
    try:
        user_data_scaled = scaler.transform(user_df)
    except ValueError as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'message': f'Scaler transformation error: {str(e)}'})
        }

    votes = []
    model_predictions = {}  # To store individual model predictions

    for model_key in model_keys:
        model_obj = s3.get_object(Bucket=bucket_name, Key=model_key)
        model = joblib.load(BytesIO(model_obj['Body'].read()))  # Use joblib for loading
        prediction = model.predict(user_data_scaled)
        votes.append(int(prediction[0]))  # Convert to native int type
        model_predictions[model_key] = int(prediction[0])  # Convert to native int type
        print(f"Model: {model_key}, Prediction: {int(prediction[0])}")

    majority_vote = 1 if votes.count(1) > votes.count(0) else 0
    result = 'Heart Disease Detected' if majority_vote == 1 else 'No Heart Disease Detected'

    print("Votes:", votes)
    print("Majority Vote Result:", result)

    # Identify missing features
    missing_features = [feature for feature in features if feature not in user_data]

    # Prepare the response
    response_message = f"Based on provided data, the result is: {result}. "
    if missing_features:
        response_message += (
            "These values were taken as normal: "
            f"{', '.join(missing_features)}. "
            "Please provide this data for a more accurate result."
        )

    response = {
        'statusCode': 200,
        'body': json.dumps({
            'message': response_message,
            'result': result,
            'model_predictions': model_predictions
        })
    }

    return response

if __name__ == "__main__":
    with open('predict.json') as f:
        event = json.load(f)
    context = {}
    result = handler(event, context)
    print(result)
