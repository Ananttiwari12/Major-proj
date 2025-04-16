import random
from fastapi import FastAPI
import psutil
from fastapi.middleware.cors import CORSMiddleware
import socket
import threading
from contextlib import asynccontextmanager
import speedtest
import time

cached_speeds={"upload":0.0, "download":0.0}

def update_speedtest():
    while True:
        try:
            st= speedtest.Speedtest()
            st.get_best_server()
            upload_speed= st.upload()
            download_speed= st.download()
            cached_speeds["upload"]= round(upload_speed/(1024*1024),2)
            cached_speeds["download"]= round(download_speed/(1024*1024),2)
        
        except Exception as e:
            print(f"speedTest Error: {e}")
        
        time.sleep(60)


@asynccontextmanager
async def lifespan(app:FastAPI):
    threading.Thread(target=update_speedtest, daemon=True).start()
    yield


app = FastAPI(lifespan=lifespan)

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
        "upload": cached_speeds["upload"],
        "download": cached_speeds["download"],
        "services": check_services()
    }

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
