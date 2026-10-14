#from langchain_openai import OpenAIEmbeddings
#from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os
load_dotenv()

#embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"),model="text-embedding-3-small")
#embeddings = GoogleGenerativeAIEmbeddings(api_key=os.getenv("GOOGLE_API_KEY"), model="models/gemini-embedding-001")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = FAISS.load_local("./Vector_DB", embeddings, allow_dangerous_deserialization=True)

from mcp.server.fastmcp import FastMCP
mcp = FastMCP("MCP Server")

@mcp.tool()
def search(question: str)-> str:
    """
    Search from the uploaded document and return the most relevant text. 
    Use this tool whenever the user asks a question relevant to the document.
    """
    
    docs = vector_store.similarity_search(question, k=5)
    if not docs:
        return "No relevant information found in the documents"
    
    texts = []
    for x in docs:
        texts.append(x.page_content)

    context = "\n".join(texts)
    return context

if __name__ == "__main__":
    mcp.run()