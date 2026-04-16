# rag_pipeline.py

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
import os
# Step 1: Load data
loader = TextLoader("data/medical.txt")
documents = loader.load()

# Step 2: Split text
splitter = CharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)
docs = splitter.split_documents(documents)

# Step 3: Embeddings (FREE)
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# Step 4: Create or Load Vector DB

if os.path.exists("vectorstore"):
    # ✅ LOAD (FAST)
    db = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)
else:
    # ❌ FIRST TIME: CREATE
    db = FAISS.from_documents(docs, embeddings)
    
    # 💾 SAVE FOR NEXT TIME
    db.save_local("vectorstore")

# Step 5: Load local LLM (Ollama)

llm = OllamaLLM(model="llama3:8b")

# Step 6: Ask function
def ask_question(query):
    retriever = db.as_retriever(search_kwargs={"k": 1})
    results = retriever.invoke(query)

    context = "\n".join([doc.page_content for doc in results])

    prompt = f"""
You are a medical assistant.
Use ONLY the context below to answer.
{context}

Question: {query}
Rules:
- Give short and clear answer
- If not in context say: I don’t know
- If serious condition → recommend doctor
- Always add: "This is general medical information."

Answer:
"""

    answer = llm.invoke(prompt)

    return answer, context