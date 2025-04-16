from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import AzureOpenAI
from dotenv import load_dotenv
import os

# Initialize FastAPI app
app = FastAPI()

#load dotenv
load_dotenv()

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (change in production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


AZURE_ENDPOINT=os.getenv("AZURE_ENDPOINT")
API_KEY=os.getenv("API_KEY")
API_VERSION=os.getenv("API_VERSION")
MODEL=os.getenv("MODEL")

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_key=API_KEY, 
    api_version=API_VERSION,
    azure_endpoint=AZURE_ENDPOINT,
)

# Define the healing prompt template
TEMPLATE = """You are an AI expert in 5G network security and intrusion detection.
Based on the observed 5G network traffic anomaly below, suggest the best mitigation strategy.

Available strategies:
1. Automated Traffic Blocking (Block IP)
2. Rate Limiting and Throttling
3. Sandbox Execution
4. Zero Trust Network Access

[Seq,Dur,sHops,dHops,SrcPkts,TotBytes,SrcBytes,Offset,sMeanPktSz,dMeanPktSz,TcpRtt,AckDat,sTtl_,dTtl_,Proto_tcp,Proto_udp,Cause_Status,State_INT]
= {anomaly}

Provide only the most appropriate strategy from the list.
"""

@app.get("/")
async def root():
    return {"message": "LLM Healing Server Running with GPT"}

@app.get("/heal")
async def heal(anomaly):
    """Receives an anomaly sample and returns a mitigation strategy."""
    try:
        # Format prompt with anomaly data
        prompt = TEMPLATE.format(anomaly=anomaly)

        # Call Azure OpenAI GPT model
        response =  client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": prompt}]    
        )

        # Extract response text
        strategy = response.choices[0].message.content
        return {strategy}
    
    except Exception as e:
        return {"error": f"LLM processing failed: {str(e)}"}
    
    
# Run using: uvicorn server2:app --host 0.0.0.0 --port 8080 --reload
