from langchain_pymupdf4llm import PyMuPDF4LLMLoader

loader = PyMuPDF4LLMLoader("document loaders/SDE Intern_JD.pdf")
documents = loader.load()

print(documents[0])