# 📡 5G Network Intrusion Detection & Healing System

An end-to-end AI-powered system to detect anomalies in 5G network traffic and automatically suggest healing/mitigation strategies using LLMs (Phi-3 or Azure GPT).

---

## 🧠 Features

- 🔍 **Real-time Anomaly Detection** using a feedforward neural network (FNN)
- 📈 **Live Monitoring Dashboard** (React + Recharts)
- 🧑‍⚕️ **AI-Based Healing** via:
  - 🤖 Local **Phi-3** model (`server.py`)
  - ☁️ **Azure OpenAI** deployment (`server2.py`)
  - **RAG** : Context retriever server
- 📡 **WebSocket**-based live communication between backend and frontend
- 📊 Preprocessed network traffic dataset included

---

## 🗂️ Project Structure

```
|
├── Combined.csv               # training dataset: can be downloaded from the provided link
|
├── backend/
│   └── server.py               # Flask server for anomaly detection using FNN
│
├── Context Retrieving Service/
│   ├── server.py               # Creates context using text-embedding-small model
|                                (deployed in Azure)
|
├── Healing Service/
│   ├── server.py               # Healing server using Phi-3 model
│   └── server2.py              # Healing server using Azure OpenAI GPT
|
├── System Health Service/
│   ├── server.py               # CPU and memory utlization, network throughput,Service checks
│
├── frontend/
│   └── network-dashboard/      # React app for live network monitoring
│
├── Preprocess_data.ipynb       # Preprocessing notebook: Utilize 'Combined.csv' to generate 'Combined_processed.csv'
|
├── neural_net.ipynb            # FNN model training code on 'Combined_processed.csv' : result- neural_net.h5
└── Makefile                    # For automation (optional)
```

---

## ⚙️ Setup Instructions

### Method 1:

### 1. Clone the repository

```bash
git clone <repo-url>
cd <repo>
pip install -r requirements.txt
```

### 2. Backend Setup (Anomaly Detection Server)

```bash
cd backend
pip install -r requirements.txt
Run uvicorn server:app --host 0.0.0.0 --port {PORT} --reload
```

> 💡 Make sure your trained FNN model is available and loaded correctly in `server.py`.

### 3. Healing Service (Choose One)

- **Phi-3 Local Healing Server:**

```bash
cd Healing\ Service
run: uvicorn server:app --host 0.0.0.0 --port {PORT} --reload
```

- **Azure Healing Server:**

(If you want to use GPT instead of phi-3)
Set your Azure API key and endpoint in `server2.py`, then:

```bash
Run uvicorn server2:app --host 0.0.0.0 --port {PORT} --reload
```

### 3. System Health Server:

```bash
cd 'System Health Service'
run: uvicorn server:app --host 0.0.0.0 --port {PORT} --reload
```

### 4. Frontend Setup

```bash
cd frontend/network-dashboard
npm install
npm run dev
```

### 5. RAG Server

```bash
cd Retriever Service
pip install -r requirements.txt
run: Run using: uvicorn server:app --host 0.0.0.0 --port {PORT} --reload

```

---

### Method 2:

Browse to the repo and open the terminal

run:

1. make run-backend
2. make run-azure-healing-server
3. make run-system-health-service
4. make run-react-code
5. make run-rag-server

## 🧪 Usage

- Start the backend (`server.py`) and one healing server (`server.py` or `server2.py`), system health service(`server.py`).
- Launch the React dashboard to visualize live traffic.
- Anomalies are detected in real-time and healing suggestions appear via LLM responses.

---

## 🏗️ Architecture

1. **FNN Model** processes network data → Detects anomaly.
2. On anomaly:
   - Sends query to either **Phi-3** or **Azure GPT** healing service.
3. Healing service sends data and query to **_ Retrive context service _**
4. **_ Retrive context service _** returns relevant context to healing server.
5. Healing service returns **mitigation strategy**.
6. Dashboard displays:
   - Real-time network flow
   - Alert + Suggested healing

---

## ✨ Future Work

- [ ] Model comparison dashboard (Phi-3 vs Azure)
- [ ] CI/CD for automatic model retraining
