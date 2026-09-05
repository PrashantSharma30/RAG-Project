from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()  # Load environment variables from .env file

loader = PyMuPDF4LLMLoader("document loaders/SDE Intern_JD.pdf")
docs = loader.load()

template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that summarizes the text."),
    ("human", "{data}")
])
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)

prompt = template.format_messages(data=docs[0].page_content)

result = llm.invoke(prompt)

print(result.content)