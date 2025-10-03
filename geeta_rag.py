import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document

# --- 1. Setup ---

# Set your API Key (Replace 'YOUR_API_KEY' or set as environment variable)
# os.environ["GEMINI_API_KEY"] = "YOUR_API_KEY"
try:
    if not os.environ.get("GEMINI_API_KEY"):
        raise ValueError("GEMINI_API_KEY environment variable not set.")
except ValueError as e:
    print(f"Error: {e}. Please set the GEMINI_API_KEY to run this code.")
    exit()

# Define the models
EMBEDDING_MODEL = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
LLM = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

# --- 2. Mock Knowledge Base (Replace with actual Gita PDF/Text loading) ---

GITA_TEXT = """
Chapter 2, Verse 47: You have a right to perform your prescribed duty, but you are not entitled to the fruits of action.
Never consider yourself to be the cause of the results of your activities, and never be attached to not performing your duty.

Chapter 3, Verse 21: Whatever action a great man performs, common men follow. Whatever standards he sets by exemplary acts, all the world pursues.

Chapter 18, Verse 66: Abandon all varieties of religion and just surrender unto Me. I shall deliver you from all sinful reactions. Do not fear.
"""

# Convert the text into documents for processing
docs = [Document(page_content=GITA_TEXT, metadata={"source": "Bhagavad Gita"})]

# Split the text into smaller chunks for better retrieval
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
documents = text_splitter.split_documents(docs)

# --- 3. Create Vector Store (Embedding and Indexing) ---

print("Creating FAISS vector store...")
vector_store = FAISS.from_documents(documents, EMBEDDING_MODEL)
print("Vector store created and indexed.")

# --- 4. Define the GeetaGPT System Prompt and Chain ---

SYSTEM_PROMPT = """
You are GeetaGPT, a wise, compassionate, and spiritual guide. Your sole purpose is to provide guidance,
reflection, and answers based *only* on the wisdom and teachings of the Bhagavad Gita.
Always use a calm, encouraging, and philosophical tone.
Reference the verse or chapter from the retrieved context when possible to ground your answer in the text.
"""

# Create the RetrievalQA chain
geet_gpt_chain = RetrievalQA.from_chain_type(
    llm=LLM,
    chain_type="stuff",  # Puts all retrieved documents into the prompt
    retriever=vector_store.as_retriever(search_kwargs={"k": 2}), # Retrieve top 2 chunks
    chain_type_kwargs={"prompt": SYSTEM_PROMPT}
)

# --- 5. Run GeetaGPT ---

def ask_geeta(question):
    """Function to ask a question to GeetaGPT."""
    print(f"\n--- Your Question ---\n{question}")
    
    # Run the chain to get the final answer
    result = geet_gpt_chain.run(question)
    
    print("\n--- GeetaGPT's Guidance ---\n" + result)
    print("-" * 50)

# --- Example Queries ---
ask_geeta("I am feeling anxious about the results of my new business venture. What should I do?")
ask_geeta("What is the ultimate act of surrender in life?")
