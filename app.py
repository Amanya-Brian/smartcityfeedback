import streamlit as st
import numpy as np
import pandas as pd
import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
import jsonlines, json

from dotenv import load_dotenv
load_dotenv()

# GROQ API
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")

key = os.getenv("GROQ_API_KEY")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on context only.
    Please provide the most accirate response based on the question.
    <context>
    {context}
    <context>
    Question: {input}

    """
)

def load_docs():
    with jsonlines.open('rwanda.jsonl') as reader:
        documents = [obj for obj in reader]

    return documents


def create_vector_embeddings():
    if "vectors" not in st.session_state:
        st.session_state.embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        st.session_state.loader=load_docs()
        st.session_state.docs=st.session_state.loader.load() 
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=300,chunk_overlap=30)



st.title("Welcome to Kigali smart city")