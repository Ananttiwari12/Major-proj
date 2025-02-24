from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Load Phi-3 Model
llm = LlamaCpp(
    model_path="D:/Intrusion_det/Healing Service/Phi-3-mini-4k-instruct-q4.gguf",
    n_gpu_layers=-1,
    max_tokens=500,
    n_ctx=2048,
    seed=42,
    verbose=True  # Set to True for debugging, can be False in production
)

# Define Prompt Template
template = """<s><|user|>

You are an AI expert in 5G network security and intrusion detection.
Based on the observed 5G network traffic anomaly provided below, suggest one appropriate mitigation strategy.

Available strategies: 
1. Automated Traffic Blocking (Block IP)
2. Rate Limiting and Throttling
3. Sandbox Execution
4. Zero Trust Network Access

The anomaly traffic data is: 

{anomaly}

Please only respond with the best suitable mitigation strategy from the list.

<|assistant|>"""
title_prompt = PromptTemplate(template=template, input_variables=["anomaly"])
title_chain = LLMChain(llm=llm, prompt=title_prompt, output_key="heal")

# Initialize FastAPI App
app = FastAPI()

# Add CORS Middleware for Cross-Origin Requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "LLM Healing Server Running"}

@app.get("/heal")
async def heal(anomaly: str = Query(..., description="Observed 5G anomaly traffic data")):
    """
    Receives an anomaly sample and returns a mitigation strategy.
    """
    try:
        # Ensure the anomaly is formatted correctly
        response = await title_chain.arun(anomaly=anomaly)
        return {"mitigation_strategy": response.strip()}
    except Exception as e:
        return {"error": f"LLM processing failed: {str(e)}"}

# Run using: uvicorn llm_server:app --host 0.0.0.0 --port 8080 --reload
