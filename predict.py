import numpy as np
import pandas as pd
from datetime import datetime
import time
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Parameters for synthetic data generation
sequence_length = 50  # LSTM input sequence length
scaler = MinMaxScaler()
fitted_scaler = False  # Flag to check if scaler is fitted

# Load the pre-trained LSTM model (assuming the model has been saved)
model = load_model('model2.h5')


# Data storage for plotting
timestamps = []
packet_sizes = []
connection_durations = []
anomaly_data = []

# Buffer to store data points for prediction
data_buffer = []

def generate_synthetic_data():
    """Generate synthetic network traffic data, occasionally introducing anomalies."""
    # Set a probability for generating an anomaly
    anomaly_probability = 0.1  # 10% chance of generating an anomaly
    is_anomaly = np.random.rand() < anomaly_probability

    if is_anomaly:
        # Generate an anomaly
        packet_size = np.random.normal(loc=5000, scale=1000)  # Abnormally large packet size
        connection_duration = np.random.normal(loc=10, scale=2)  # Abnormally long duration
        print("Anomaly generated!")
    else:
        # Generate normal data
        packet_size = np.random.normal(loc=500, scale=100)  # Normal packet size
        connection_duration = np.random.normal(loc=2, scale=0.5)  # Normal duration

    timestamp = datetime.now()

    # Create a single data point
    return {
        "timestamp": timestamp,
        "packet_size": max(0, packet_size),  # Ensure non-negative
        "connection_duration": max(0, connection_duration)  # Ensure non-negative
    }


def fit_scaler(data):
    """Fit the MinMaxScaler with the initial batch of data."""
    global fitted_scaler
    if not fitted_scaler:
        # Fit the scaler on initial batch of data
        initial_data = np.array([[data['packet_size'], data['connection_duration']]])
        scaler.fit(initial_data)
        fitted_scaler = True
        print("Scaler fitted on initial data.")

def predict_anomaly(buffer):
    """Predict if the incoming data points are an anomaly using the buffer."""
    # Scale the buffered data
    buffer_scaled = scaler.transform(buffer)
    
    # Prepare the input shape for LSTM (batch_size, time_steps, features)
    input_data = np.reshape(buffer_scaled, (1, sequence_length, 2))  # Assuming 2 features

    # Predict using the LSTM model
    reconstruction = model.predict(input_data)
    # Calculate reconstruction error
    error = np.mean(np.square(reconstruction - input_data))
    
    # Log reconstruction error for analysis
    print(f"Reconstruction error: {error}")

    # Define a threshold for anomaly detection
    threshold = 0.05  # Consider adjusting this value based on your analysis
    if error > threshold:
        print("Anomaly is predicted")
    return error > threshold


def plot_data(is_anomaly):
    """Plot the network traffic data and anomalies."""
    plt.clf()  # Clear the current figure
    plt.subplot(2, 1, 1)

    # Change color based on anomaly detection
    color = 'red' if is_anomaly else 'blue'
    plt.plot(timestamps, packet_sizes, label='Packet Size (bytes)', color=color)
    plt.xlabel('Time')
    plt.ylabel('Packet Size')
    plt.title('Network Packet Size Over Time')
    plt.xticks(rotation=45)
    plt.legend()

    # Plot anomalies
    if len(anomaly_data) == len(timestamps):  # Check if both lists have the same length
        plt.subplot(2, 1, 2)
        plt.plot(timestamps, anomaly_data, label='Anomaly Detected', color='red', marker='o', linestyle='None')
        plt.axhline(0, color='grey', lw=0.5, ls='--')
        plt.xlabel('Time')
        plt.ylabel('Anomaly Flag (1=Anomaly, 0=No Anomaly)')
        plt.title('Anomaly Detection Over Time')
        plt.xticks(rotation=45)
        plt.ylim(-0.1, 1.1)
        plt.legend()
    else:
        print(f"Warning: Mismatch in lengths of timestamps ({len(timestamps)}) and anomaly_data ({len(anomaly_data)})")

    plt.tight_layout()
    plt.pause(0.1)  # Pause to update the plot



def main():
    print("Starting network monitoring...")
    plt.ion()  # Turn on interactive mode for real-time plotting
    previous_anomaly_state = False  # Track the previous anomaly state

    while True:
        # Generate synthetic data
        new_data = generate_synthetic_data()
        
        # Fit the scaler with the first generated data point
        fit_scaler(new_data)

        # Store generated data for plotting
        timestamps.append(new_data['timestamp'])
        packet_sizes.append(new_data['packet_size'])
        connection_durations.append(new_data['connection_duration'])

        # Append the new data to the buffer for prediction
        data_buffer.append([new_data['packet_size'], new_data['connection_duration']])
        
        # Initialize anomaly detection flag
        is_anomaly = False
        
        # Check if we have enough data points for prediction
        if len(data_buffer) >= sequence_length:
            # Use the last 'sequence_length' data points for prediction
            buffer_for_prediction = data_buffer[-sequence_length:]
            is_anomaly = predict_anomaly(buffer_for_prediction)
            
            # Append to anomaly data; log anomaly detection
            anomaly_data.append(1 if is_anomaly else 0)
            
            if is_anomaly:
                print(f"[ALERT] Anomaly detected at {new_data['timestamp']}: {new_data['packet_size']} bytes, {new_data['connection_duration']} seconds")
            else:
                # If the current prediction is normal and the previous state was an anomaly, 
                # we should reset the previous state flag
                if previous_anomaly_state:
                    print(f"[INFO] Anomaly state cleared at {new_data['timestamp']}.")

        else:
            # If not enough data points, append 0 to anomaly_data
            anomaly_data.append(0)

        # Update the previous anomaly state for the next iteration
        previous_anomaly_state = is_anomaly

        # Update the plot with the anomaly status
        plot_data(is_anomaly)
        
        # Wait for some time before generating the next data point (simulate real-time monitoring)
        time.sleep(1)  # Adjust sleep time as needed


if __name__ == "__main__":
    main()
