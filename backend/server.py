from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import asyncio
import random
from datetime import datetime

# Load Model and Scaler
model = tf.keras.models.load_model("../neural_net_model.h5")
scaler = joblib.load("scaler.pkl")

# Load Test Data (assuming first column is an index, last column is label)
df = pd.read_csv("../TestData2.csv")
X_test = df.iloc[:, 1:-1].values
y_test = df.iloc[:, -1].values

# Scale X_test
X_test_scaled = scaler.transform(X_test)

# Initialize FastAPI
app = FastAPI()

# Add CORS middleware to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store active WebSocket connections
active_connections = set()

@app.get("/")
def home():
    return {"message": "Network Monitoring API Running"}

@app.websocket("/ws/monitor")
async def websocket_monitor(websocket: WebSocket):
    """WebSocket to stream real-time monitoring data."""
    await websocket.accept()
    active_connections.add(websocket)
    try:
        while True:
            # Choose a random normal sample (where y_test == 0)
            
            sample= X_test[y_test==0]
            
            if len(sample) == 0:
                await websocket.send_json({"error": "No normal samples found."})
                return
            
            sample= random.choice(sample).reshape(1,-1)
            normal_sample= scaler.transform(sample)
            prediction = model.predict(normal_sample)[0][0]
            
            print(sample)
            
            
            # The below code is written for debugging
            # if prediction > 0.5:
            #     prediction= random.uniform(0.0,0.49)
            
                
            result = {
                "timestamp": datetime.now().isoformat(),
                "probability": float(prediction),
                "anomaly": float(prediction > 0.9)
            }
            await websocket.send_json(result)
            await asyncio.sleep(1)  # Send update every 1 second
            
    except WebSocketDisconnect:
        active_connections.discard(websocket)
        print("WebSocket client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        active_connections.discard(websocket)

@app.post("/introduce_anomaly")
async def introduce_anomaly():
    """
    Force an anomaly by selecting an index from y_test where y_test == 1.
    Broadcast the anomaly data to all active WebSocket clients.
    """
    anomaly_indices = np.where(y_test == 1)[0]
    if len(anomaly_indices) == 0:
        return {"error": "No anomalies found in dataset"}
    index = random.choice(anomaly_indices)
    sample = X_test_scaled[index].reshape(1, -1)
    prediction = model.predict(sample)[0][0]
    
    non_scaled_sample= X_test[index].reshape(1,-1)
    
    # print(non_scaled_sample)
    
    # The below code is written for debugging
    
    # if prediction < 0.5:
    #     prediction= random.uniform(0.51,1.0)
    
    
    anomaly_result = {
        "timestamp": datetime.now().isoformat(),
        "probability": float(prediction),
        "anomaly": 1.0
    }
    # Broadcast anomaly data
    for connection in list(active_connections):
        try:
            await connection.send_json(anomaly_result)
        except Exception as e:
            print(f"Error sending anomaly data: {e}")
            active_connections.discard(connection)
    return anomaly_result

# Run using: uvicorn server:app --host 0.0.0.0 --port 8000 --reload
