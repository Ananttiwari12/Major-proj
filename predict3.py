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

        # Load the model first to safely get the input shape
        self.model = self._load_model(model_path)
        
        if self.model:
            # Store model's expected feature count
            self.n_features = self.model.input_shape[-1]  # This should only be accessed after the model is loaded
            logger.info(f"Model expects {self.n_features} features")
        else:
            logger.error("Model is not loaded correctly. Exiting initialization.")
            raise ValueError("Model loading failed")

        # Initialize deques for data storage
        max_points = 1000
        self.timestamps: Deque = deque(maxlen=max_points)
        self.latencies: Deque[float] = deque(maxlen=max_points)
        self.bandwidths: Deque[float] = deque(maxlen=max_points)
        self.packet_losses: Deque[float] = deque(maxlen=max_points)
        self.anomaly_scores: Deque[float] = deque(maxlen=max_points)
        
        # Buffer for sequence data
        self.data_buffer: Deque[List[float]] = deque(maxlen=self.sequence_length)
        
        # Initialize metrics
        self.initialize_metrics()
        
        # Anomaly detection parameters
        self.threshold = 0.1  # Adjusted threshold for anomaly detection
        
        # Setup plotting
        plt.ion()
        self.fig, self.axes = plt.subplots(4, 1, figsize=(15, 15))
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
            if input_shape[-1] != self.n_features:
                raise ValueError(f"Model expects {input_shape[-1]} features, but dataset has {self.n_features}")
            
            return model
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            return None  # Return None if loading fails


    def _load_model(self, model_path: str):
        """Load the pre-trained model with error handling."""
        try:
            model = load_model(model_path)
            input_shape = model.input_shape
            output_shape = model.output_shape
            
            logger.info(f"Model input shape: {input_shape}")
            logger.info(f"Model output shape: {output_shape}")
            
            # Check if input and output shapes are compatible with our monitoring needs
            if input_shape[-1] != self.n_features:
                raise ValueError(f"Model expects {input_shape[-1]} features, but dataset has {self.n_features}")
                
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
            self.latencies.append(initial_data["latency"])
            self.bandwidths.append(initial_data["bandwidth"])
            self.packet_losses.append(initial_data["packet_loss"])
            self.anomaly_scores.append(0.0)
            
            # Initialize buffer with all features
            self.data_buffer.append([
                initial_data["latency"],
                initial_data["bandwidth"],
                initial_data["jitter"],
                initial_data["throughput"],
                initial_data["signal_strength"],
                initial_data["connection_density"]
            ])

    def generate_synthetic_data(self, normal: bool = True) -> Dict[str, float]:
        """Generate synthetic network traffic data."""
        if normal:
            latency = np.random.normal(loc=20, scale=5)
            bandwidth = np.random.normal(loc=100, scale=10)
            jitter = np.random.normal(loc=1, scale=0.5)
            throughput = np.random.normal(loc=50, scale=5)
            packet_loss = np.random.uniform(0, 1)
            signal_strength = np.random.normal(loc=-70, scale=10)
            connection_density = np.random.randint(100, 1000)
        else:
            # Anomalous data
            latency = np.random.normal(loc=100, scale=20)
            bandwidth = np.random.normal(loc=10, scale=2)
            jitter = np.random.normal(loc=10, scale=3)
            throughput = np.random.normal(loc=10, scale=2)
            packet_loss = np.random.uniform(5, 10)
            signal_strength = np.random.normal(loc=-120, scale=5)
            connection_density = np.random.randint(1000, 5000)

        return {
            "timestamp": datetime.now(),
            "latency": max(0, float(latency)),
            "bandwidth": max(0, float(bandwidth)),
            "jitter": max(0, float(jitter)),
            "throughput": max(0, float(throughput)),
            "packet_loss": float(packet_loss),
            "signal_strength": float(signal_strength),
            "connection_density": float(connection_density)
        }

    def predict_next_values(self, buffer: List[List[float]]) -> tuple:
        """Predict next latency and bandwidth values."""
        try:
            buffer_array = np.array(buffer)
            
            if not hasattr(self.scaler, 'scale_'):
                self.scaler.fit(buffer_array)
            
            buffer_scaled = self.scaler.transform(buffer_array)
            input_data = np.reshape(buffer_scaled, (1, self.sequence_length, self.n_features))
            
            # Get prediction
            prediction = self.model.predict(input_data, verbose=0)[0]
            
            # Calculate anomaly score based on prediction error
            next_actual = buffer_array[-1][:2]  # Get last actual latency and bandwidth
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

            # Plot latency
            self.axes[0].plot(timestamps, list(self.latencies)[-plot_size:], 'b-', label='Actual')
            self.axes[0].set_title('End-to-End Latency Over Time')
            self.axes[0].set_ylabel('Latency (ms)')
            self.axes[0].legend()

            # Plot bandwidth
            self.axes[1].plot(timestamps, list(self.bandwidths)[-plot_size:], 'b-', label='Actual')
            self.axes[1].set_title('Bandwidth Over Time')
            self.axes[1].set_ylabel('Bandwidth (Mbps)')
            self.axes[1].legend()

            # Plot packet loss
            self.axes[2].plot(timestamps, list(self.packet_losses)[-plot_size:], 'b-', label='Actual')
            self.axes[2].set_title('Packet Loss Rate Over Time')
            self.axes[2].set_ylabel('Packet Loss (%)')
            self.axes[2].legend()

            # Plot anomaly scores
            self.axes[3].plot(timestamps, list(self.anomaly_scores)[-plot_size:], 'r-')
            self.axes[3].axhline(y=self.threshold, color='g', linestyle='--', label='Threshold')
            self.axes[3].set_title('Anomaly Score')
            self.axes[3].set_ylabel('Score')
            self.axes[3].legend()

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
                self.latencies.append(new_data['latency'])
                self.bandwidths.append(new_data['bandwidth'])
                self.packet_losses.append(new_data['packet_loss'])
                
                # Update buffer with all features
                self.data_buffer.append([
                    new_data["latency"],
                    new_data["bandwidth"],
                    new_data["jitter"],
                    new_data["throughput"],
                    new_data["signal_strength"],
                    new_data["connection_density"]
                ])
                
                # Make prediction if we have enough data
                if len(self.data_buffer) >= self.sequence_length:
                    pred_latency, pred_bandwidth, anomaly_score = self.predict_next_values(list(self.data_buffer))
                    
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
        monitor = NetworkMonitor(model_path='model7.h5')
        monitor.run()
    except Exception as e:
        logger.error(f"Failed to start monitoring: {e}")