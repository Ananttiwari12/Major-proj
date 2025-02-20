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
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# class NetworkMonitor:
import joblib

class NetworkMonitor:
    def __init__(self, model_path: str, test_data_path: str):
        """Initialize the Network Monitor."""
        self.model = self._load_model(model_path)
        self.n_features = self.model.input_shape[-1]
        
        # Load the saved scaler
        self.scaler = self._load_scaler("scaler.pkl")

        # Load X_test and y_test
        self.X_test, self.y_test = self._load_test_data(test_data_path)
        self.test_index = 0  # Index to iterate over test data
        max_points = 1000
        self.timestamps: Deque = deque(maxlen=max_points)
        self.total_bytes: Deque[float] = deque(maxlen=max_points)      # TotBytes
        self.connection_durations: Deque[float] = deque(maxlen=max_points)  # Dur
        self.anomaly_probs: Deque[float] = deque(maxlen=max_points)
        
        # Anomaly detection threshold (for example, probability > 0.5 indicates an anomaly)
        self.threshold = 0.5
        
        # Setup plotting: Three subplots for TotBytes, Dur, and anomaly probability.
        plt.ion()
        self.fig, self.axes = plt.subplots(3, 1, figsize=(15, 12))
        self.fig.tight_layout(pad=3.0)
        
    def _load_test_data(self, test_data_path: str):
        """Load X_test and y_test from a CSV file."""
        try:
            df = pd.read_csv(test_data_path)
            
            # Assuming the last column is 'y' (labels)
            X_test = df.iloc[:, 1:-1].values
            y_test = df.iloc[:, -1].values

            logger.info(f"Loaded test data: {X_test.shape} samples")
            return X_test, y_test
        except Exception as e:
            logger.error(f"Error loading test data: {e}")
            raise
        
    def _load_scaler(self, scaler_path: str):
        """Load the saved StandardScaler"""
        try:
            scaler = joblib.load(scaler_path)
            logger.info("Scaler loaded successfully.")
            return scaler
        except Exception as e:
            logger.error(f"Failed to load scaler: {e}")
            raise

        
    def _load_model(self, model_path: str):
        """Load the pre-trained model with error handling."""
        try:
            model = load_model(model_path)
            input_shape = model.input_shape
            output_shape = model.output_shape
            
            logger.info(f"Model input shape: {input_shape}")
            logger.info(f"Model output shape: {output_shape}")
            
            # Check if the model's input shape matches our expectation.
            # (For example, if your dataset has 18 features, input_shape[-1] should be 18.)
            # Adjust the number below as necessary.
            if input_shape[-1] not in [18, 19]:
                raise ValueError(f"Model expects {input_shape[-1]} features, but 18 or 19 are required.")
                
            return model
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            raise

    def generate_synthetic_data(self, normal: bool = True) -> Dict[str, Any]:
        """Generate synthetic network traffic data for a single flow.
        
        The returned dictionary includes keys corresponding to your dataset features.
        Here we assume a feature order (example for 19 features):
        ['Seq', 'Dur', 'sHops', 'dHops', 'SrcPkts', 'TotBytes', 'SrcBytes', 'Offset',
         'sMeanPktSz', 'dMeanPktSz', 'TcpRtt', 'AckDat', 'sTtl_', 'dTtl_', 
        'Proto_tcp', 'Proto_udp', 'Cause_Status', 'State_INT']
        """
        if normal:
            return {
                "timestamp": datetime.now(),
                "Seq": np.random.randint(1, 100000),
                "Dur": np.random.normal(loc=1.0, scale=0.2),
                "sHops": np.random.randint(1, 10),
                "dHops": np.random.randint(1, 10),
                "SrcPkts": np.random.randint(10, 30),
                "TotBytes": np.random.normal(loc=1500, scale=300),
                "SrcBytes": np.random.normal(loc=800, scale=200),
                "Offset": np.random.randint(0, 10),
                "sMeanPktSz": np.random.normal(loc=50, scale=10),
                "dMeanPktSz": np.random.normal(loc=50, scale=10),
                "TcpRtt": np.random.normal(loc=100, scale=20),
                "AckDat": np.random.randint(0, 2),
                "sTtl_": np.random.randint(50, 70),
                "dTtl_": np.random.randint(50, 70),
                "Proto_tcp": 1,
                "Proto_udp": 0,
                "Cause_Status": 0,
                "State_INT": 0
            }
        else:
            return {
                "timestamp": datetime.now(),
                "Seq": np.random.randint(1, 100000),
                "Dur": np.random.normal(loc=10.0, scale=2.0),
                "sHops": np.random.randint(1, 10),
                "dHops": np.random.randint(1, 10),
                "SrcPkts": np.random.randint(30, 60),
                "TotBytes": np.random.normal(loc=5000, scale=1000),
                "SrcBytes": np.random.normal(loc=3000, scale=500),
                "Offset": np.random.randint(0, 20),
                "sMeanPktSz": np.random.normal(loc=100, scale=20),
                "dMeanPktSz": np.random.normal(loc=100, scale=20),
                "TcpRtt": np.random.normal(loc=300, scale=50),
                "AckDat": np.random.randint(0, 2),
                "sTtl_": np.random.randint(20, 40),
                "dTtl_": np.random.randint(20, 40),
                "Proto_tcp": 0,
                "Proto_udp": 1,
                "Cause_Status": 1,
                "State_INT": 1
            }
            
    def extract_features(self, data: Dict[str, Any]) -> List[float]:
        """Extract features from the data dictionary in the required order.
        
        Adjust the order here to match your dataset.
        """
        return [
            data["Seq"],
            data["Dur"],
            data["sHops"],
            data["dHops"],
            data["SrcPkts"],
            data["TotBytes"],
            data["SrcBytes"],
            data["Offset"],
            data["sMeanPktSz"],
            data["dMeanPktSz"],
            data["TcpRtt"],
            data["AckDat"],
            data["sTtl_"],
            data["dTtl_"],
            data["Proto_tcp"],
            data["Proto_udp"],
            data["Cause_Status"],
            data["State_INT"]
        ]
        

    # def predict_sample(self, features: List[float]) -> tuple:
    #     """Predict anomaly probability for a single sample.
        
    #     Returns:
    #         predicted_prob: The model's output probability (e.g., probability of attack).
    #         anomaly_score: (Optional) Here we can use the same as predicted probability,
    #                        or compute a difference from a baseline if available.
    #     """
    #     try:
    #         sample = np.array(features).reshape(1, self.n_features)
    #         if not self.scaler_fitted:
    #             # Fit the scaler on the first sample (or ideally on training data)
    #             self.scaler.fit(sample)
    #             self.scaler_fitted = True
    #         sample_scaled = self.scaler.transform(sample)
    #         prediction = self.model.predict(sample_scaled, verbose=0)[0]
    #         predicted_prob = prediction[0]  # Assuming a single output neuron with sigmoid
    #         # For simplicity, we use predicted probability as the anomaly score
    #         anomaly_score = predicted_prob  
    #         return predicted_prob, anomaly_score
    #     except Exception as e:
    #         logger.error(f"Error in prediction: {e}")
    #         raise

    def predict_sample(self, features: List[float]) -> tuple:
        """Predict anomaly probability for a single sample."""
        try:
            sample = np.array(features).reshape(1, -1)
            sample_scaled = self.scaler.transform(sample)  # Use saved scaler
            prediction = self.model.predict(sample_scaled, verbose=0)[0]
            predicted_prob = prediction[0]
            return predicted_prob, predicted_prob  # Using predicted prob as anomaly score
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            raise

    def update_plot(self):
        """Update the real-time visualization plots."""
        try:
            for ax in self.axes:
                ax.clear()

            plot_size = 100  # Last 100 points for plotting
            timestamps = list(self.timestamps)[-plot_size:]

            # Plot TotBytes (Total Bytes)
            self.axes[0].plot(timestamps, list(self.total_bytes)[-plot_size:], 'b-', label='Actual TotBytes')
            self.axes[0].set_title('Total Bytes Over Time')
            self.axes[0].set_ylabel('TotBytes')
            self.axes[0].legend()

            # Plot Duration (Dur)
            self.axes[1].plot(timestamps, list(self.connection_durations)[-plot_size:], 'b-', label='Actual Duration')
            self.axes[1].set_title('Duration Over Time')
            self.axes[1].set_ylabel('Duration (s)')
            self.axes[1].legend()

            # Plot anomaly probabilities
            self.axes[2].plot(timestamps, list(self.anomaly_probs)[-plot_size:], 'r-')
            self.axes[2].axhline(y=self.threshold, color='g', linestyle='--', label='Threshold')
            self.axes[2].set_title('Anomaly Probability')
            self.axes[2].set_ylabel('Probability')
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
        """Main monitoring loop using real test data."""
        logger.info("Starting network monitoring...")
        
        try:
            while self.test_index < len(self.X_test):
                # Get the next test sample
                sample = self.X_test[self.test_index]
                label = self.y_test[self.test_index]  # 1 = anomaly, 0 = normal
                
                # Increment test index
                self.test_index += 1

                # Extract features from test data
                self.timestamps.append(datetime.now())
                self.total_bytes.append(sample[5])  # Assuming TotBytes is at index 5
                self.connection_durations.append(sample[1])  # Assuming Dur is at index 1

                # Predict anomaly probability
                predicted_prob, anomaly_score = self.predict_sample(sample)

                # Store anomaly probability
                self.anomaly_probs.append(anomaly_score)

                # Log if actual anomaly and detected anomaly match
                if label == 1:
                    logger.warning(f"True Anomaly! Predicted Probability: {anomaly_score:.4f}")
                elif anomaly_score > self.threshold:
                    logger.warning(f"False Positive! Probability: {anomaly_score:.4f}")

                # Update plot
                self.update_plot()

                time.sleep(0.5)  # Adjust for real-time monitoring

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
        monitor = NetworkMonitor(model_path="neural_net_model.h5", test_data_path="TestData.csv")
        monitor.run()
    except Exception as e:
        logger.error(f"Failed to start monitoring: {e}")
