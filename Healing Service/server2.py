from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import AzureOpenAI
from dotenv import load_dotenv
import os
import json

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
TEMPLATE = """You are an embedded AI security module within a 5G network intrusion detection system. Your task is to provide IMMEDIATE, AUTOMATED mitigation commands that will be directly executed via API calls to the 5G core components (AMF, SMF, UPF) without human intervention.

Network Anomaly Details:
[Sequence: {Seq}, Duration: {Dur} sec, Source Hops: {sHops}, Destination Hops: {dHops}, 
Source Packets: {SrcPkts}, Total Bytes: {TotBytes}, Source Bytes: {SrcBytes}, 
Offset: {Offset}, Source Mean Packet Size: {sMeanPktSz}, Destination Mean Packet Size: {dMeanPktSz}, 
TCP RTT: {TcpRtt}, ACK Data Ratio: {AckDat}, Source TTL: {sTtl_}, Destination TTL: {dTtl_},
Protocol_is_TCP :{Proto_tcp}, Protocol_is_UDP: {Proto_udp}, 
Cause Status: {Cause_Status}, State: {State_INT}]

YOUR RESPONSE MUST CONTAIN ONLY THESE SECTIONS:
1. Threat Assessment: <Single sentence anomaly classification>
2. Mitigation Command: <Specific 5G network API command>
3. Parameters:
   a. Target: <AMF|SMF|UPF|gNB identifier>
   b. Action: <rate_limit|block_flow|isolate_slice|reroute|log_only>
   c. Duration: <seconds>
   d. Severity: <low|medium|high|critical>
4. Verification: <API endpoint to check mitigation status>
5. Fallback: <Alternative command if primary fails>

IMPORTANT CONSTRAINTS:
- All mitigations MUST be executable via the 5G core API without human intervention
- Do NOT suggest installing new software, changing configurations, or other non-automated responses
- Focus on commands that can be implemented IMMEDIATELY (under 5 seconds)
- Prioritize service continuity - use graduated response based on threat severity
- Commands must follow 5G network protocols and standards (3GPP TS 23.501, TS 23.502)
- Responses must be specific to 5G networks (AMF, SMF, UPF, gNB components)
"""

@app.get("/")
async def root():
    return {"message": "LLM Healing Server Running with GPT"}

@app.get("/heal")
async def heal(anomaly):
    """Receives an anomaly sample and returns a mitigation strategy."""
    try:
        anomaly_data= json.loads(anomaly)
        
        # Format prompt with anomaly data
        prompt = TEMPLATE.format(**anomaly_data)

        # Call Azure OpenAI GPT model
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a 5G network security expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )

        # Extract response text
        strategy = response.choices[0].message.content
        return {strategy}
    except json.JSONDecodeError:
        return {"error": "Invalid JSON format for anomaly data"}
    except Exception as e:
        return {"error": f"LLM processing failed: {str(e)}"}
    
# Run using: uvicorn server2:app --host 0.0.0.0 --port 8080 --reload