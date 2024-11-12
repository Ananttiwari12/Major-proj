import numpy as np
import pandas as pd
from datetime import datetime
import time
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from collections import deque
from typing import Dict, List, Any, Deque
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NetworkMonitor:
    def __init__(self, model_path: str, sequence_length: int = 50):
        """Initialize the Network Monitor with configuration parameters."""
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler()
        self.model = self._load_model(model_path)
        
        # Store model's expected feature count
        self.n_features = self.model.input_shape[-1]
        logger.info(f"Model expects {self.n_features} features")
        
        # Initialize deques for data storage
        max_points = 1000
        self.timestamps: Deque = deque(maxlen=max_points)
        self.packet_sizes: Deque[float] = deque(maxlen=max_points)
        self.connection_durations: Deque[float] = deque(maxlen=max_points)
        self.predicted_packet_sizes: Deque[float] = deque(maxlen=max_points)
        self.predicted_connection_durations: Deque[float] = deque(maxlen=max_points)
        self.anomaly_scores: Deque[float] = deque(maxlen=max_points)
        
        # Buffer for sequence data
        self.data_buffer: Deque[List[float]] = deque(maxlen=self.sequence_length)
        
        # Initialize metrics
        self.initialize_metrics()
        
        # Anomaly detection parameters
        self.threshold = 0.1  # Adjusted threshold for anomaly detection
        
        # Setup plotting
        plt.ion()
        self.fig, self.axes = plt.subplots(3, 1, figsize=(15, 12))
        self.fig.tight_layout(pad=3.0)

    def _load_model(self, model_path: str):
        """Load the pre-trained model with error handling."""
        try:
            model = load_model(model_path)
            input_shape = model.input_shape
            output_shape = model.output_shape
            
            logger.info(f"Model input shape: {input_shape}")
            logger.info(f"Model output shape: {output_shape}")
            
            # Check if input and output shapes are compatible with our monitoring needs
            if input_shape[-1] != 6:  # Expecting 6 features
                raise ValueError(f"Model expects {input_shape[-1]} features, need 6")
                
            return model
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            raise

    def initialize_metrics(self):
        """Initialize all metrics with empty values."""
        # Generate synthetic initial data
        initial_data = self.generate_synthetic_data(normal=True)
        for _ in range(self.sequence_length):
            self.timestamps.append(datetime.now())
            self.packet_sizes.append(initial_data["packet_size"])
            self.connection_durations.append(initial_data["connection_duration"])
            self.predicted_packet_sizes.append(initial_data["packet_size"])
            self.predicted_connection_durations.append(initial_data["connection_duration"])
            self.anomaly_scores.append(0.0)
            
            # Initialize buffer with all features
            self.data_buffer.append([
                initial_data["packet_size"],
                initial_data["connection_duration"],
                initial_data["packet_inter_arrival_time"],
                initial_data["SYN_packets"],
                initial_data["packet_size_connection_ratio"],
                initial_data["SYN_packet_ratio"]
            ])

    def generate_synthetic_data(self, normal: bool = True) -> Dict[str, float]:
        """Generate synthetic network traffic data."""
        if normal:
            packet_size = np.random.normal(loc=500, scale=100)
            connection_duration = np.random.normal(loc=2, scale=0.5)
            packet_inter_arrival_time = np.random.normal(loc=0.02, scale=0.005)
            SYN_packets = np.random.randint(0, 10)
            packet_size_connection_ratio = np.random.normal(loc=250, scale=50)
            SYN_packet_ratio = np.random.uniform(0, 1)
        else:
            # Anomalous data
            packet_size = np.random.normal(loc=5000, scale=1000)
            connection_duration = np.random.normal(loc=10, scale=2)
            packet_inter_arrival_time = np.random.normal(loc=0.1, scale=0.02)
            SYN_packets = np.random.randint(10, 20)
            packet_size_connection_ratio = np.random.normal(loc=500, scale=100)
            SYN_packet_ratio = np.random.uniform(0.8, 1)

        return {
            "timestamp": datetime.now(),
            "packet_size": max(0, float(packet_size)),
            "connection_duration": max(0.1, float(connection_duration)),
            "packet_inter_arrival_time": max(0.001, float(packet_inter_arrival_time)),
            "SYN_packets": float(SYN_packets),
            "packet_size_connection_ratio": max(0, float(packet_size_connection_ratio)),
            "SYN_packet_ratio": min(1, max(0, float(SYN_packet_ratio)))
        }

    def predict_next_values(self, buffer: List[List[float]]) -> tuple:
        """Predict next packet size and connection duration."""
        try:
            buffer_array = np.array(buffer)
            
            if not hasattr(self.scaler, 'scale_'):
                self.scaler.fit(buffer_array)
            
            buffer_scaled = self.scaler.transform(buffer_array)
            input_data = np.reshape(buffer_scaled, (1, self.sequence_length, self.n_features))
            
            # Get prediction
            prediction = self.model.predict(input_data, verbose=0)[0]
            
            # Calculate anomaly score based on prediction error
            next_actual = buffer_array[-1][:2]  # Get last actual packet size and duration
            anomaly_score = np.mean(np.square(prediction - next_actual))
            
            return prediction[0], prediction[1], anomaly_score
            
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            raise

    def update_plot(self):
        """Update the real-time visualization plots."""
        try:
            for ax in self.axes:
                ax.clear()

            # Get last 100 points for plotting
            plot_size = 100
            timestamps = list(self.timestamps)[-plot_size:]

            # Plot packet sizes
            self.axes[0].plot(timestamps, list(self.packet_sizes)[-plot_size:], 'b-', label='Actual')
            self.axes[0].plot(timestamps, list(self.predicted_packet_sizes)[-plot_size:], 'r--', label='Predicted')
            self.axes[0].set_title('Packet Size Over Time')
            self.axes[0].set_ylabel('Packet Size (bytes)')
            self.axes[0].legend()

            # Plot connection durations
            self.axes[1].plot(timestamps, list(self.connection_durations)[-plot_size:], 'b-', label='Actual')
            self.axes[1].plot(timestamps, list(self.predicted_connection_durations)[-plot_size:], 'r--', label='Predicted')
            self.axes[1].set_title('Connection Duration Over Time')
            self.axes[1].set_ylabel('Duration (s)')
            self.axes[1].legend()

            # Plot anomaly scores
            self.axes[2].plot(timestamps, list(self.anomaly_scores)[-plot_size:], 'r-')
            self.axes[2].axhline(y=self.threshold, color='g', linestyle='--', label='Threshold')
            self.axes[2].set_title('Anomaly Score')
            self.axes[2].set_ylabel('Score')
            self.axes[2].legend()

            for ax in self.axes:
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True)

            plt.tight_layout()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

        except Exception as e:
            logger.error(f"Error updating plot: {e}", exc_info=True)

    def run(self):
        """Main monitoring loop."""
        logger.info("Starting network monitoring...")
        
        try:
            while True:
                # Generate new data (10% chance of anomaly)
                is_normal = np.random.rand() > 0.1
                new_data = self.generate_synthetic_data(normal=is_normal)
                
                # Update data storage
                self.timestamps.append(new_data['timestamp'])
                self.packet_sizes.append(new_data['packet_size'])
                self.connection_durations.append(new_data['connection_duration'])
                
                # Update buffer with all features
                self.data_buffer.append([
                    new_data["packet_size"],
                    new_data["connection_duration"],
                    new_data["packet_inter_arrival_time"],
                    new_data["SYN_packets"],
                    new_data["packet_size_connection_ratio"],
                    new_data["SYN_packet_ratio"]
                ])
                
                # Make prediction if we have enough data
                if len(self.data_buffer) >= self.sequence_length:
                    pred_packet_size, pred_duration, anomaly_score = self.predict_next_values(list(self.data_buffer))
                    
                    self.predicted_packet_sizes.append(pred_packet_size)
                    self.predicted_connection_durations.append(pred_duration)
                    self.anomaly_scores.append(anomaly_score)
                    
                    if anomaly_score > self.threshold:
                        logger.warning(f"Anomaly detected! Score: {anomaly_score:.4f}")
                
                self.update_plot()
                time.sleep(0.5)  # Reduced sleep time for more responsive visualization

        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user.")
        except Exception as e:
            logger.error(f"Unexpected error in monitoring loop: {e}")
            raise
        finally:
            plt.ioff()
            plt.close('all')

if __name__ == "__main__":
    try:
        monitor = NetworkMonitor(model_path='model4.h5')
        monitor.run()
    except Exception as e:
        logger.error(f"Failed to start monitoring: {e}")