from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()  # Load environment variables from .env file

loader = PyMuPDF4LLMLoader("document loaders/MT2025069_MT2025091_MT2025013_VR_Report.pdf")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=150,
    chunk_overlap=30,
)

chunks = text_splitter.split_documents(docs)

template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that summarizes the text."),
    ("human", "{data}")
])
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)

print(len(chunks))
#prompt = template.format_messages(data=chunks)

#result = llm.invoke(prompt)

#print(result.content)