import numpy as np
from datetime import datetime
import time
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from collections import deque
from typing import Dict, List, Any, Deque
import logging
import matplotlib.dates as mdates


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NetworkMonitor:
    def __init__(self, model_path: str):
        """Initialize the Network Monitor with configuration parameters."""
        self.scaler = MinMaxScaler()
        self.model = self._load_model(model_path)

        self.all_features = ['Seq', 'Dur', 'sHops', 'dHops', 'SrcPkts', 'TotBytes', 'SrcBytes', 'Offset', 'sMeanPktSz', 'dMeanPktSz', 'TcpRtt', 'AckDat', 'sTtl_', 'dTtl_', 'Proto_tcp', 'Proto_udp', 'Cause_Status', 'State_INT']

        # Initialize deques for data storage
        max_points = 1000
        self.timestamps: Deque = deque(maxlen=max_points)
        self.anomaly_scores: Deque[float] = deque(maxlen=max_points)

        # Store only the latest feature vector
        self.data_buffer: Deque[List[float]] = deque(maxlen=1)

        # Initialize metrics
        self.initialize_metrics()
        
        # Anomaly detection parameters
        self.threshold = 0.5  # Adjusted threshold for anomaly detection
        
        # Setup plotting
        plt.ion()  # Turn on interactive mode
        self.fig = plt.figure(figsize=(15, 5))  # Create a figure
        self.ax = plt.gca()  # Get the current axes
        plt.tight_layout(pad=3.0)  # Adjust spacing


    def _load_model(self, model_path: str):
        """Load the pre-trained model with error handling."""
        try:

            model = load_model(model_path, compile=True)

            input_shape = model.input_shape
            output_shape = model.output_shape
            
            logger.info(f"Model input shape: {input_shape}")
            logger.info(f"Model output shape: {output_shape}")
            
            return model
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            raise

    def initialize_metrics(self):
        """Initialize all metrics with empty values."""
        initial_data = self.generate_synthetic_data(normal=True)

        self.timestamps.append(datetime.now())
        
        self.anomaly_scores.append(0.0)
        
        # Clear previous buffer and store only the latest data point
        self.data_buffer.clear() 
        # Initialize buffer with all features
        self.data_buffer.append([initial_data[feature] for feature in self.all_features])


    def generate_synthetic_data(self, normal: bool = True) -> dict:
        """Generate synthetic network traffic data."""
        if normal:
            seq = np.random.randint(1, 100)
            dur = np.random.normal(loc=2, scale=0.5)
            src_pkts = np.random.randint(1, 20)
            tot_bytes = np.random.randint(100, 10000)
            src_bytes = np.random.randint(50, tot_bytes)
            dst_bytes = tot_bytes - src_bytes
            offset = np.random.randint(100, 2000)
            s_mean_pkt_sz = src_bytes / src_pkts if src_pkts > 0 else 0
            d_mean_pkt_sz = dst_bytes / max(1, np.random.randint(1, 10))
            tcp_rtt = np.random.normal(loc=0.2, scale=0.05)
            ack_dat = np.random.choice([0, 1])
            s_ttl = np.random.randint(50, 128)
            d_ttl = np.random.randint(50, 128)
            attack_type = 0  # Normal
            proto_tcp = np.random.choice([True, False], p=[0.7, 0.3])
            proto_udp = not proto_tcp
            cause_status = False
            state_int = np.random.choice([True, False], p=[0.2, 0.8])
        else:
            seq = np.random.randint(100, 200)
            dur = np.random.normal(loc=10, scale=2)
            src_pkts = np.random.randint(20, 100)
            tot_bytes = np.random.randint(10000, 50000)
            src_bytes = np.random.randint(5000, tot_bytes)
            dst_bytes = tot_bytes - src_bytes
            offset = np.random.randint(2000, 5000)
            s_mean_pkt_sz = src_bytes / src_pkts if src_pkts > 0 else 0
            d_mean_pkt_sz = dst_bytes / max(1, np.random.randint(1, 20))
            tcp_rtt = np.random.normal(loc=1.0, scale=0.2)
            ack_dat = np.random.choice([0, 1])
            s_ttl = np.random.randint(20, 60)
            d_ttl = np.random.randint(20, 60)
            attack_type = 1  # Anomalous
            proto_tcp = np.random.choice([True, False], p=[0.3, 0.7])
            proto_udp = not proto_tcp
            cause_status = True
            state_int = np.random.choice([True, False], p=[0.8, 0.2])
        
        return {
            "timestamp": datetime.now(),
            "Seq": int(seq),
            "Dur": max(0.1, float(dur)),
            "sHops": np.random.randint(0, 20),
            "dHops": np.random.randint(0, 20),
            "SrcPkts": float(src_pkts),
            "TotBytes": float(tot_bytes),
            "SrcBytes": float(src_bytes),
            "Offset": float(offset),
            "sMeanPktSz": float(s_mean_pkt_sz),
            "dMeanPktSz": float(d_mean_pkt_sz),
            "TcpRtt": float(tcp_rtt),
            "AckDat": float(ack_dat),
            "sTtl_": float(s_ttl),
            "dTtl_": float(d_ttl),
            "Attack Type_": int(attack_type),
            "Proto_tcp": bool(proto_tcp),
            "Proto_udp": bool(proto_udp),
            "Cause_Status": bool(cause_status),
            "State_INT": bool(state_int)
        }

    def predict_next_values(self, buffer: List[List[float]]) -> Dict[str, float]:
        """Predict next values for all features."""
        try:
            buffer_array = np.array(buffer[-1]).reshape(1, -1)  # Take only the last feature vector
        
            if not hasattr(self.scaler, 'scale_'):
                self.scaler.fit(buffer_array)

            buffer_scaled = self.scaler.transform(buffer_array) 
            input_data = buffer_scaled  # Already in shape (1, num_features)

            # Get prediction
            prediction = self.model.predict(input_data, verbose=0)[0]

            return prediction[0]
            
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            raise

    def update_plot(self):
        """Update the real-time visualization plot."""
        try:
            plt.clf()  # Clear the previous plot

            # Get last 100 points for plotting
            plot_size = 100
            timestamps = list(self.timestamps)[-plot_size:]
            anomaly_scores = list(self.anomaly_scores)[-plot_size:]

            # Plot anomaly scores
            plt.plot(timestamps, anomaly_scores, 'r-', label='Anomaly Score')
            plt.axhline(y=self.threshold, color='b', linestyle='--', label='Threshold')

            plt.title('Anomaly Score')
            plt.xlabel('Time')
            plt.ylabel('Score')
            plt.xticks(rotation=45)
            plt.grid(True)
            plt.legend()

            plt.tight_layout()
            plt.draw()
            plt.pause(0.01)  # Small pause for real-time updates

        except Exception as e:
            logger.error(f"Error updating plot: {e}", exc_info=True)

    def run(self):
        """Main monitoring loop."""
        logger.info("Starting network monitoring...")
        
        try:
            while True:
                # Generate new data (10% chance of anomaly)
                is_normal = np.random.rand() > 0.1
                new_data = self.generate_synthetic_data(normal = is_normal)
                
                self.timestamps.append(datetime.now())

                self.data_buffer.clear()  # Remove previous values
                # Update buffer with all features
                self.data_buffer.append([new_data[feature] for feature in self.all_features])
                
                # Make prediction if we have enough data
                if len(self.data_buffer) > 0:  
                    anomaly_score = self.predict_next_values(list(self.data_buffer))
                    
                    # for feature, pred_value in pred_values.items():
                    #     self.predicted_values[feature].append(pred_value)
                    
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
        monitor = NetworkMonitor(model_path='neural_net_model.h5')
        monitor.run()
    except Exception as e:
        logger.error(f"Failed to start monitoring: {e}")