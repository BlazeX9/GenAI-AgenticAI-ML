#from langchain_openai import OpenAIEmbeddings
#from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from pypdf import PdfReader
import configuration
import os
load_dotenv()

def pdf_vector_store(file):
    reader = PdfReader(file)
    pdf_text = ""

    for x in reader.pages:
        pdf_text += (x.extract_text() or "") + "\n"

    pdf_documents = []
    pdf_documents.append(pdf_text)

    #embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"), model="text-embedding-3-small")
    #embeddings = GoogleGenerativeAIEmbeddings(api_key=os.getenv("GOOGLE_API_KEY"), model="models/gemini-embedding-001")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = configuration.PDF_CHUNK_SIZE, 
        chunk_overlap = configuration.PDF_CHUNK_OVERLAP
    )

    documents = []
    for i in pdf_documents:
        chunks = splitter.split_text(i)
        documents.extend(chunks)

    FAISS.from_texts(documents, embeddings).save_local("./Vector_DB")