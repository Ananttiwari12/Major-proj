import random
from fastapi import FastAPI
import psutil
from fastapi.middleware.cors import CORSMiddleware
import socket

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/system_metrics")
def get_metrics():
    return {
        "cpu": psutil.cpu_percent(),
        "memory": psutil.virtual_memory().percent,
        "upload": get_network_throughput("upload"),
        "download": get_network_throughput("download"),
        "services": check_services()
    }

def get_network_throughput(direction):
    return round(random.uniform(10, 100), 1)

def check_services():
    return {
        "detector": "up" if is_port_open("localhost", 8000) else "down",
        "mitigator": "up" if is_port_open("localhost", 8080) else "down",
        "database": "up" if is_port_open("localhost", 5432) else "down"
    }

def is_port_open(host, port):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            s.connect((host, port))
        return True
    except:
        return False
    
    
# Run using: uvicorn server:app --host 0.0.0.0 --port 6001 --reload
