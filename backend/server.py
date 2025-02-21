# from fastapi import FastAPI, WebSocket, WebSocketDisconnect
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# import joblib
# import asyncio
# # from langchain.chat_models import ChatOpenAI
# # from langchain.schema import HumanMessage
# import random

# # Load Model and Scaler
# model = tf.keras.models.load_model("neural_net_model.h5") 
# scaler = joblib.load("scaler.pkl")

# # Load Test Data
# df = pd.read_csv("TestData.csv")
# X_test = df.iloc[:, 1:-1].values
# y_test = df.iloc[:, -1].values

# # Scale X_test
# X_test_scaled = scaler.transform(X_test)

# # Initialize FastAPI
# app = FastAPI()

# # LangChain with Microsoft Phi-3 SLM (if available)
# # llm = ChatOpenAI(model_name="microsoft/phi-3", temperature=0.3)

# # Store connected clients
# active_connections = set()

# @app.get("/")
# def home():
#     return {"message": "Network Monitoring API Running"}

# @app.websocket("/ws/monitor")
# async def websocket_monitor(websocket: WebSocket):
#     """WebSocket to send real-time monitoring data."""
#     await websocket.accept()
#     active_connections.add(websocket)

#     try:
#         while True:
#             # Select a normal sample (y_test == 0)
#             normal_samples = X_test_scaled[y_test == 0]
#             if len(normal_samples) == 0:
#                 await websocket.send_json({"error": "No normal samples found."})
#                 return

#             sample = normal_samples[random.randint(0, len(normal_samples) - 1)].reshape(1, -1)
#             prediction = model.predict(sample)

#             result = {
#                 "probability": float(prediction[0][0]),
#                 "anomaly": prediction[0][0] > 0.9
#             }

#             await websocket.send_json(result)
#             await asyncio.sleep(1)  # 1-second interval

#     except WebSocketDisconnect:
#         print("Client disconnected")
#     except Exception as e:
#         print(f"WebSocket error: {e}")
#     finally:
#         active_connections.discard(websocket)

# @app.post("/introduce_anomaly")
# async def introduce_anomaly():
#     """Forces an anomaly by selecting an index from y_test where y=1."""
#     anomaly_indices = np.where(y_test == 1)[0]  # Get indices of anomalies
#     if len(anomaly_indices) == 0:
#         return {"error": "No anomalies found in dataset"}

#     index = random.choice(anomaly_indices)
#     sample = X_test_scaled[index].reshape(1, -1)
#     prediction = model.predict(sample)

#     anomaly_result = {
#         "probability": float(prediction[0][0]),
#         "anomaly": True
#     }

#     # Send anomaly data to all WebSocket clients
#     for connection in active_connections.copy():
#         try:
#             await connection.send_json(anomaly_result)
#         except Exception as e:
#             print(f"Error sending anomaly data: {e}")
#             active_connections.discard(connection)  # Remove disconnected clients

#     return anomaly_result

# # @app.post("/get_prevention")
# # def get_prevention():
# #     """Fetches anomaly prevention methods from Microsoft Phi-3 SLM via LangChain."""
# #     response = llm([HumanMessage(content="A network anomaly has been detected. Suggest prevention strategies.")])
# #     return {"prevention_methods": response.content}



# from fastapi import FastAPI, WebSocket, WebSocketDisconnect
# import numpy as np
# import pandas as pd
# import joblib
# import tensorflow as tf
# import asyncio
# import random
# from datetime import datetime

# # Load Model and Scaler
# model = tf.keras.models.load_model("neural_net_model.h5")
# scaler = joblib.load("scaler.pkl")

# # Load Test Data
# df = pd.read_csv("TestData.csv")
# X_test = df.iloc[:, 1:-1].values
# y_test = df.iloc[:, -1].values
# X_test_scaled = scaler.transform(X_test)

# # FastAPI App
# app = FastAPI()

# # Store active WebSocket connections
# connections = set()

# @app.websocket("/ws/monitor")
# async def websocket_endpoint(websocket: WebSocket):
#     await websocket.accept()
#     connections.add(websocket)
#     try:
#         while True:
#             # Pick a normal sample and predict
#             normal_samples = X_test_scaled[y_test == 0]
#             sample = random.choice(normal_samples).reshape(1, -1)
#             prediction = model.predict(sample)[0][0]
#             result = {
#                 "timestamp": datetime.now().isoformat(),
#                 "probability": float(prediction),
#                 "anomaly": float(prediction > 0.9)
#             }
#             await websocket.send_json(result)
#             await asyncio.sleep(1)  # Stream every second
#     except WebSocketDisconnect:
#         connections.remove(websocket)

# @app.post("/introduce_anomaly")
# async def introduce_anomaly():
#     anomaly_samples = np.where(y_test == 1)[0]
#     if not anomaly_samples.any():
#         return {"error": "No anomalies found in dataset"}
    
#     index = random.choice(anomaly_samples)
#     sample = X_test_scaled[index].reshape(1, -1)
#     prediction = model.predict(sample)[0][0]
#     anomaly_result = {
#         "timestamp": datetime.now().isoformat(),
#         "probability": float(prediction),
#         "anomaly": True
#     }
#     print(anomaly_result)
#     for connection in connections:
#         await connection.send_json(anomaly_result)
    
#     return anomaly_result


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
model = tf.keras.models.load_model("neural_net_model.h5")
scaler = joblib.load("scaler.pkl")

# Load Test Data (assuming first column is an index, last column is label)
df = pd.read_csv("TestData.csv")
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
            normal_samples = X_test_scaled[y_test == 0]
            if len(normal_samples) == 0:
                await websocket.send_json({"error": "No normal samples found."})
                return
            sample = random.choice(normal_samples).reshape(1, -1)
            prediction = model.predict(sample)[0][0]
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
