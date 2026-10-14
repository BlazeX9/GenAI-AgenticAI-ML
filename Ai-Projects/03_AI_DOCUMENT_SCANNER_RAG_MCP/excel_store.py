#from langchain_openai import OpenAIEmbeddings
#from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import pandas as pd
import configuration
import os
load_dotenv()

def excel_vector_store(file):
    if file.name.lower().endswith(".csv"):
        reader = pd.read_csv(file)
    else:
        reader = pd.read_excel(file)

    excel_text = ""

    for _, row in reader.iterrows():
        row_text = "\n".join(
            [f"{column}: {value}" for column, value in row.items()]
        )
        excel_text += row_text + "\n\n"

    excel_documents = []
    excel_documents.append(excel_text)

    #embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"), model="text-embedding-3-small")
    #embeddings = GoogleGenerativeAIEmbeddings(api_key=os.getenv("GOOGLE_API_KEY"), model="models/gemini-embedding-001")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = configuration.EXCEL_CHUNK_SIZE, 
        chunk_overlap = configuration.EXCEL_CHUNK_OVERLAP
    )
    
    documents = []
    for i in excel_documents:
        chunks = splitter.split_text(i)
        documents.extend(chunks)
    
    FAISS.from_texts(documents, embeddings).save_local("./Vector_DB")