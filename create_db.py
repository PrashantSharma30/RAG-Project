#load pdf 
#split into chunks 
#create the embeddings 
#store into chroma 
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma 

from dotenv import load_dotenv

load_dotenv()

data = PyPDFLoader("document loaders/SDE Intern_JD.pdf")
docs = data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 250,
    chunk_overlap = 10
)

chunks = splitter.split_documents(docs)

embedding_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

vectorstore = Chroma.from_documents(
    documents= chunks,
    embedding=embedding_model,
    persist_directory="chroma_db"
)