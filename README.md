
# Smart-Home-Energy-Management-System-with-Machine-Learning

This project involves sending power consumption data from a Raspberry Pi to AWS IoT Core, storing the data in DynamoDB using AWS Lambda, and using Amazon SageMaker to train and deploy an LSTM (Long Short-Term Memory) model for prediction.

## Architecture Overview

1. **Raspberry Pi Setup:**
   - Collects power consumption data using an ACS712 sensor.
   - Sends data securely to AWS IoT Core for processing.

2. **AWS IoT Core:**
   - Receives data from Raspberry Pi via MQTT protocol.
   - Routes data to AWS Lambda for processing.

3. **AWS Lambda:**
   - Processes incoming data from AWS IoT Core.
   - Stores data into DynamoDB.

4. **DynamoDB:**
   - Stores power consumption data in a NoSQL format.

5. **Amazon SageMaker:**
   - Retrieves data from DynamoDB.
   - Trains an LSTM model to predict power consumption patterns.
   - Deploys the trained model for real-time predictions.

## Requirements

- Raspberry Pi (models with GPIO pins)
- ACS712 Current Sensor (or similar sensor)
- AWS Account with access to AWS IoT Core, Lambda, DynamoDB, and SageMaker
- Python 3.x installed on Raspberry Pi
- RPi.GPIO library for GPIO operations (if using Raspberry Pi GPIO)
- boto3 library for AWS SDK operations on Raspberry Pi

## Setup Instructions

### 1. Raspberry Pi Setup

#### Hardware Setup
- Connect the ACS712 sensor to Raspberry Pi GPIO pins.
- Ensure proper wiring and power connections according to the sensor specifications.

#### Software Setup
- Install Python 3.x and required libraries (RPi.GPIO, boto3) on Raspberry Pi.

### 2. AWS IoT Core Setup

#### Thing and Topic Setup
1. Create an AWS IoT Thing:
   - Go to AWS IoT Core console.
   - Navigate to "Manage" -> "Things".
   - Click on "Create" and follow the steps to create a new IoT Thing. Note down the Thing name.

2. Create an IoT Core Policy:
   - In AWS IoT Core console, navigate to "Secure" -> "Policies".
   - Click on "Create" and define a policy with necessary permissions (e.g., publish/subscribe to a specific topic).
   - Attach the policy to your IoT Thing.

### 3. AWS Lambda and DynamoDB Setup

#### AWS Lambda Function
1. Create an AWS Lambda Function:
   - Go to AWS Lambda console.
   - Click on "Create function" and choose Python as the runtime.
   - Use the provided sample code (`lambda_function.py`) to process data and store it in DynamoDB.
    ```python
        import json
        import boto3
        from decimal import Decimal
        from datetime import datetime
        import logging

        # Configure the logging module
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        
        table_name = 'PowerConsumptionTable'

        def lambda_handler(event, context):
             client = boto3.client('dynamodb')
             # Log the entire event to understand its structure
             logger.info(f"Received event: {json.dumps(event)}")
        
             timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M')
             entry_no = event['entryno']
             power_consumption = Decimal(str(event['Power_Consumption']))
            
             logger.info(f"Parsed message: {event}")
            
             # Insert the item into DynamoDB
             response = client.put_item(
                   TableName = table_name,
                    Item={
                          'entry_no': {'N': str(entry_no)},  # Number type
                          'timestamp': {'S': timestamp},  # String type
                          'Power_Consumption': {'N': str(power_consumption)}
                        }
                   )
            
             logger.info('Done Inputting')
             return 0

    ```
2. Configure Lambda Triggers:
   - Set AWS IoT Core as the trigger for the Lambda function to process incoming MQTT messages.

#### DynamoDB Setup
1. Create a DynamoDB Table:
   - In AWS DynamoDB console, click on "Create table".
   - Define the table name (YourDynamoDBTableName), primary key, and other attributes as per your data schema.


### 4. Amazon SageMaker Setup

#### Data Preparation
1. Extract Data from DynamoDB:
   - Use Python SDK (boto3) to extract data from DynamoDB.
   - Prepare data for training by performing necessary feature engineering and data preprocessing steps.

#### Model Training and Deployment
1. Train an LSTM Model:
   - Use Amazon SageMaker's built-in algorithms or custom scripts to train an LSTM model on the prepared data.
   - Tune hyperparameters and validate model performance using SageMaker capabilities.
		#### Prepare the SageMaker Notebook
		1. Create a new SageMaker Notebook instance in the AWS console.
		2. Open the Jupyter Notebook interface.
		3. Create a new Python 3 notebook.
		4. Copy and paste the following code into the notebook:

		```python
		#!/usr/bin/env python
		# coding: utf-8

		# Install required libraries
		!pip install boto3 pandas tensorflow

		import boto3
		import pandas as pd
		import numpy as np
		from sklearn.preprocessing import MinMaxScaler
		import tensorflow as tf
		from tensorflow.keras.models import Sequential
		from tensorflow.keras.layers import LSTM, Dense
		import matplotlib.pyplot as plt

		# Initialize DynamoDB client
		client = boto3.client('dynamodb')

		# Fetch all data from DynamoDB table
		table_name = 'PowerConsumptionTable'
		response = client.scan(TableName=table_name)
		data = response['Items']

		# Convert to DataFrame
		df = pd.DataFrame(data)
		df['timestamp'] = pd.to_datetime(df['timestamp'].apply(lambda x: x['S']))
		df['entry_no'] = df['entry_no'].apply(lambda x: int(x['N']))
		df['Power_Consumption'] = df['Power_Consumption'].apply(lambda x: float(x['N']))

		# Sort by timestamp
		df.sort_values(by='timestamp', inplace=True)
		df.set_index('timestamp', inplace=True)

		print(df)

		# Select features
		features = df[['Power_Consumption']].values

		# Normalize the features
		scaler = MinMaxScaler()
		scaled_features = scaler.fit_transform(features)

		# Create sequences for LSTM
		def create_sequences(data, seq_length):
		    xs = []
		    ys = []
		    for i in range(len(data) - seq_length):
		        x = data[i:i + seq_length]
		        y = data[i + seq_length]
		        xs.append(x)
		        ys.append(y)
		    return np.array(xs), np.array(ys)

		seq_length = 4  # Number of time steps
		X, y = create_sequences(scaled_features, seq_length)

		# Split data into training and testing sets
		train_size = int(len(X) * 0.8)
		X_train, X_test = X[:train_size], X[train_size:]
		y_train, y_test = y[:train_size], y[train_size:]

		# Define the LSTM model
		model = Sequential()
		model.add(LSTM(50, activation='relu', input_shape=(seq_length, 1)))
		model.add(Dense(1))
		model.compile(optimizer='adam', loss='mse')

		# Train the model
		history = model.fit(X_train, y_train, epochs=20, batch_size=1, validation_split=0.2)

		# Plot training history
		plt.plot(history.history['loss'], label='train')
		plt.plot(history.history['val_loss'], label='validation')
		plt.legend()
		plt.show()

		# Make predictions
		y_pred = model.predict(X_test)

		# Inverse transform to get actual values
		y_test_actual = scaler.inverse_transform(y_test)
		y_pred_actual = scaler.inverse_transform(y_pred)

		# Plot the results
		plt.figure(figsize=(10, 6))
		plt.plot(y_test_actual, label='Actual Power Consumption')
		plt.plot(y_pred_actual, label='Predicted Power Consumption')
		plt.xlabel('Time')
		plt.ylabel('Power Consumption')
		plt.legend()
		plt.show()
		```
2. Deploy the Trained Model:
	-   Save & Deploy the trained LSTM model as an endpoint in Amazon SageMaker.
	-   Integrate the endpoint with your application for real-time predictions.
		```python
		model.save('/tmp/lstm_power_consumption_model.h5')
		```
	- Use SageMaker's built-in TensorFlow serving container to deploy the model:
		```python
			import sagemaker
			from sagemaker.tensorflow import TensorFlowModel

			sagemaker_session = sagemaker.Session()
			role = sagemaker.get_execution_role()

			model_data = sagemaker_session.upload_data(
			    path='/tmp/lstm_power_consumption_model.h5',
			    key_prefix='model'
			)

			tensorflow_model = TensorFlowModel(
			    model_data=model_data,
			    role=role,
			    framework_version='2.3.0',
			    entry_point='inference.py'
			)

			predictor = tensorflow_model.deploy(
			    initial_instance_count=1,
			    instance_type='ml.t2.medium'
			)
		```
	- Create an `inference.py` file in the same directory as your notebook with the following content:
		```python
				import json
				import tensorflow as tf
				import numpy as np
				
				def model_fn(model_dir):
				    model = tf.keras.models.load_model(model_dir + '/lstm_power_consumption_model.h5')
				    return model

				def input_fn(request_body, request_content_type):
				    if request_content_type == 'application/json':
				        input_data = json.loads(request_body)
				        return np.array(input_data)
				    else:
				        raise ValueError("Unsupported content type: {}".format(request_content_type))

				def predict_fn(input_data, model):
				    predictions = model.predict(input_data)
				    return predictions

				def output_fn(predictions, content_type):
				    if content_type == 'application/json':
				        return json.dumps(predictions.tolist())
				    raise ValueError("Unsupported content type: {}".format(content_type))
		```
## Usage

1. **Raspberry Pi:**
   - If you're working with the ACS712 sensor to collect real-time data: 
	   - Run the Python script `Main.py` on Raspberry Pi to start sending data to AWS IoT Core. 
	- If you're working with an existing dataset: 
		- Use the `PowerConsumption.py` script to process and send the data to AWS IoT Core.

2. **AWS Lambda and DynamoDB:**
   - Ensure Lambda function (`lambda_function.py`) is correctly triggered by AWS IoT Core.
   - Monitor DynamoDB (YourDynamoDBTableName) for incoming data and storage performance.

3. **Amazon SageMaker:**
   - Use SageMaker Notebook instances or SDK to interact with DynamoDB data.
   - Train and deploy LSTM model using SageMaker capabilities.

## Troubleshooting

- **AWS IAM Permissions:** Verify Lambda and SageMaker roles have necessary permissions.
- **Data Flow:** Ensure data flows correctly from Raspberry Pi to DynamoDB and SageMaker.
- **Sensor Calibration:** Adjust sensor readings and data preprocessing steps as needed.