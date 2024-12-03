import os
import streamlit as st
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
import json

# Step 1: Load Data and Create Vector Database
def load_data(file_path):
    """Load JSON data from a file."""
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]

# def prepare_documents(data):
#     """Split content into chunks and create LangChain-compatible documents."""
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
#     documents = []
#     for entry in data:
#         content = entry.get("content", "")
#         title = entry.get("title", "Unknown Title")
#         if content:
#             chunks = text_splitter.split_text(content)
#             for chunk in chunks:
#                 documents.append({"page_content": chunk, "metadata": {"title": title}})
#     return documents

def prepare_documents(data):
    """Split content into chunks and create LangChain-compatible documents."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    documents = []
    
    for entry in data:
        content = entry.get("content", "")
        title = entry.get("title", "Unknown Title")
        metadata = entry.get("metadata", {})
        
        if content:
            chunks = text_splitter.split_text(content)
            for chunk in chunks:
                documents.append({
                    "page_content": "always here", #chunk, 
                    "metadata": {
                        "title": title, 
                        **metadata  # Include any additional metadata
                    }
                })
    
    return documents


def build_vectorstore(docs, persist_directory="db_folder"):
    """Build and persist a vector database."""
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding_function=embedding_model,
        persist_directory=persist_directory
    )
    vectordb.persist()
    return vectordb

# Step 2: Initialize the Chat Model and QA Pipeline
def initialize_qa_chain(vectorstore):
    """Initialize the ChatGroq model and RetrievalQA pipeline."""
    model = ChatGroq(model="gemma-7b-it", groq_api_key=os.getenv("GROQ_API_KEY"))
    retriever = vectorstore.as_retriever()
    return RetrievalQA.from_chain_type(
        llm=model,
        retriever=retriever,
        return_source_documents=True
    )

# Step 3: Build the Streamlit UI
def run_app(qa_chain):
    """Streamlit UI for interacting with the chatbot."""
    st.title("RAG Chatbot with ChatGroq")
    st.write("Ask questions about the data you scraped!")

    query = st.text_input("Enter your question:")
    if query:
        result = qa_chain({"query": query})
        answer = result["result"]
        source_docs = result["source_documents"]

        # Display the answer
        st.write("### Answer:")
        st.write(answer)

        # Display source documents
        st.write("### Sources:")
        for doc in source_docs:
            title = doc.metadata.get("title", "Unknown Title")
            content = doc.page_content
            st.write(f"#### {title}")
            st.write(content)

# Main Execution
if __name__ == "__main__":
    # Load and prepare data
    DATA_FILE_PATH = "rwanda.jsonl"  # Replace with your file path
    data = load_data(DATA_FILE_PATH)

    docs = prepare_documents(data)

    # Build the vector database
    vectorstore = build_vectorstore(docs)

    # Initialize QA chain
    qa_chain = initialize_qa_chain(vectorstore)

    # Run the Streamlit app
    run_app(qa_chain)
