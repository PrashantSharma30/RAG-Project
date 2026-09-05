from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()  # Load environment variables from .env file

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)
result = llm.invoke("Hello, how are you?")

if isinstance(result.content, list):
    print("".join(
        block["text"]
        for block in result.content
        if block.get("type") == "text"
    ))
else:
    print(result.content)