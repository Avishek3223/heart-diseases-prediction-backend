import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import boto3
import joblib
from io import BytesIO, StringIO
from sklearn.preprocessing import StandardScaler

def create_ann():
    model = Sequential()
    model.add(Dense(32, input_dim=13, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def handler(event, context):
    # S3 bucket and file key
    bucket_name = event['bucket']
    file_key = event['key']

    # Download the preprocessed dataset from S3
    s3 = boto3.client('s3')
    try:
        obj = s3.get_object(Bucket=bucket_name, Key=file_key)
        data = obj['Body'].read().decode('utf-8')
        df = pd.read_csv(StringIO(data))
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'message': f'Error downloading or reading data from S3: {str(e)}'})
        }

    # Split the data into training and testing sets
    features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    X = df[features]
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train multiple models
    models = {
        'logistic_regression': LogisticRegression(max_iter=1000),
        'random_forest': RandomForestClassifier(),
        'svm': SVC(probability=True),
        'ann': KerasClassifier(model=create_ann, epochs=50, batch_size=10, verbose=0),
        'xgboost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        'gradient_boosting': GradientBoostingClassifier(),
        'extra_trees': ExtraTreesClassifier(),
        'knn': KNeighborsClassifier(),
        'lightgbm': LGBMClassifier(),
        'catboost': CatBoostClassifier(verbose=0),
    }

    accuracies = {}

    for name, model in models.items():
        try:
            # Train the model
            model_X_train, model_X_test = X_train_scaled, X_test_scaled
            if name == 'catboost':
                model_X_train, model_X_test = X_train, X_test

            if name == 'ann':
                model.fit(model_X_train, y_train, epochs=50, batch_size=10, verbose=0)
                model.model.save('/tmp/ann_model.h5')  # Save to a temporary location
                s3.upload_file('/tmp/ann_model.h5', bucket_name, f'models/{name}_model.h5')
            else:
                model.fit(model_X_train, y_train)
                y_pred = model.predict(model_X_test)
                accuracy = accuracy_score(y_test, y_pred)
                accuracies[name] = accuracy

                # Save the model to a binary stream
                model_buffer = BytesIO()
                joblib.dump(model, model_buffer)
                model_buffer.seek(0)
                # Upload the model to S3
                model_key = f'models/{name}_model.pkl'
                s3.put_object(Bucket=bucket_name, Key=model_key, Body=model_buffer.getvalue())

        except Exception as e:
            print(f"Error training or saving model {name}: {str(e)}")

    # Save the scaler to a binary stream
    scaler_buffer = BytesIO()
    joblib.dump(scaler, scaler_buffer)
    scaler_buffer.seek(0)

    # Upload the scaler to S3
    scaler_key = 'models/scaler.pkl'
    s3.put_object(Bucket=bucket_name, Key=scaler_key, Body=scaler_buffer.getvalue())

    return {
        'statusCode': 200,
        'body': json.dumps({'message': 'Models and scaler trained successfully', 'accuracies': accuracies})
    }

if __name__ == "__main__":
    # For local testing
    with open('test-event-train.json') as f:
        event = json.load(f)
    context = {}
    result = handler(event, context)
    print(result)