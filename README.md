## Data Flow Diagram

![heart diseases drawio](https://github.com/user-attachments/assets/63f332d3-c228-4554-bd8a-c6cf43d64ecc)


This diagram illustrates the flow of data through the different stages of the project, from data preprocessing to model training and prediction.

# Heart Disease Prediction Project

This project aims to predict the likelihood of heart disease using various machine learning models. The application is designed to preprocess data, train models, and make predictions based on user-provided inputs. The project is deployed using AWS Lambda and stores data and models in AWS S3.

## Project Structure

- **Preprocessing**: Cleans and scales the input data.
- **Training**: Trains multiple machine learning models and saves them to S3.
- **Prediction**: Loads the trained models and makes predictions based on new data.

## Features

- Supports multiple machine learning models including Logistic Regression, Random Forest, SVM, XGBoost, LightGBM, and CatBoost.
- Uses AWS S3 for data and model storage.
- Provides a RESTful API for making predictions.

## Setup Instructions

### Prerequisites

- Python 3.8+
- AWS account with S3 bucket access
- Serverless framework

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/heart-disease-prediction.git
    cd heart-disease-prediction
    ```

2. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Install the Serverless framework plugins:
    ```bash
    serverless plugin install -n serverless-python-requirements
    serverless plugin install -n serverless-layers
    ```

### Deployment

1. Configure AWS credentials for the Serverless framework.

2. Deploy the project:
    ```bash
    serverless deploy
    ```

### Local Testing

You can test the functions locally using the Serverless framework:

- **Preprocess Data**:
    ```bash
    serverless invoke local --function preprocess --path mocks/preprocess.json
    ```

- **Train Models**:
    ```bash
    serverless invoke local --function train_models --path mocks/train_models.json
    ```

- **Predict**:
    ```bash
    serverless invoke local --function predict --path mocks/predict.json
    ```

## Usage

### Preprocessing Data

The preprocessing function downloads a dataset from S3, fills missing values, scales the data, and uploads the preprocessed data back to S3.

### Training Models

The training function downloads the preprocessed data from S3, splits it into training and testing sets, trains multiple machine learning models, and uploads the trained models back to S3.

### Making Predictions

The prediction function loads the trained models from S3, scales the user input data, and provides a prediction indicating the likelihood of heart disease.

## Prediction Example

Below is an example prediction made by the system:

![Prediction Example](images/prediction_example.png)

## Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss changes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
