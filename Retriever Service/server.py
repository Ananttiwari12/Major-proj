import os
from azure.ai.inference import EmbeddingsClient
from azure.core.credentials import AzureKeyCredential
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.embeddings import Embeddings
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

app = FastAPI()
load_dotenv()

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (change in production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

endpoint = os.getenv("EMB_ENDPOINT")
model_name = os.getenv("EMB_MODEL")
emb_key=os.getenv("EMB_KEY")
file_path= os.getenv("FILE_PATH")

embeddings_client = EmbeddingsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(emb_key)
)

class CustomAzureEmbeddings(Embeddings):
    def __init__(self, client, model_name):
        self.client = client
        self.model_name = model_name
    
    def embed_documents(self, texts):
        """Get embeddings for a list of texts."""
        if not texts:
            return []
            
        response = self.client.embed(
            input=texts,
            model=self.model_name
        )
        
        return [item.embedding for item in response.data]
    
    def embed_query(self, text):
        """Get embedding for a single text."""
        response = self.client.embed(
            input=[text],
            model=self.model_name
        )
        return response.data[0].embedding
    
# Initialize our custom embeddings class
embeddings = CustomAzureEmbeddings(embeddings_client, model_name)
loader = PyPDFLoader(file_path)
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
docs = text_splitter.split_documents(documents)
vectorstore = InMemoryVectorStore(embedding=embeddings)
vectorstore.add_documents(docs)
retriever = vectorstore.as_retriever()

@app.get("/ping")
def ping():
    return {"status": "RAG service running...."}


@app.get("/get_context")
def get_context():
    """Receives request for context...."""
    relevant_docs= retriever.invoke("5G intrusion detection data features")
    context= "\n\n".join([doc.page_content for doc in relevant_docs])
    return {"context": context}
      
# Run using: uvicorn server:app --host 0.0.0.0 --port 5050 --reload