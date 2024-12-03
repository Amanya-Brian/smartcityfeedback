from langchain_community.llms import Ollama
import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="Kigali Smart City Guide", page_icon="🏙️")
st.title("🏙️ Kigali Smart City Guide")

## Langsmith Tracking
# os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_API_KEY"]=st.secrets["LANGCHAIN_API_KEY"] # for streamlit deployment
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]="Final Kigali Smart City Assistant"

# Instantiate model
## Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant for anyone looking for information about Kigali city, especially investment opportunities. "
                   "Please respond to the user queries. You do not know anything else apart from Kigali."),
        ("user", "Question: {question}"),
        ("system", "Additional Context: {local}")
    ]
)

chroma_db_path = "/chroma_db"
vectorstore = Chroma(persist_directory=chroma_db_path, embedding_function=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))


def generate_local_response(question):
    """
    Retrieve relevant context from ChromaDB.
    """
    retriever = vectorstore.as_retriever()
    docs = retriever.get_relevant_documents(question)
    # Combine retrieved documents into a single string for local context
    response = "\n".join([doc.page_content for doc in docs])
    return response

def generate_response(question, local, llm, temperature, max_tokens):
    """
    Generate a response augmented with local query results.
    """
    # Initialize LLM
    llm = Ollama(model=llm)
    output_parser = StrOutputParser()
    
    # Build the chain dynamically with the local context
    chain = prompt | llm | output_parser
    
    # Pass both question and local information to the chain
    answer = chain.invoke({'question': question, 'local': local})
    return answer

# Example Question
# question = "Any 10 tourist areas in Kigali?"
# local_query_result = local
temperature = 0.2
max_tokens = 150
llm = "gemma2:2b" #"mistral"

# Generate response with augmented local context
# response = generate_response(question, local_query_result, llm, temperature, max_tokens)
# print(local)
# print()
# print(response)

## Main interface for user input
st.write("Go ahead and ask any question about Kigali")
user_input=st.text_input("You:")

# generate responses
if user_input :
    with st.spinner("Processing..."):
        local_query_result = generate_local_response(user_input)
        response=generate_response(user_input,local_query_result,llm,temperature,max_tokens)
        st.write(response)
else:
    st.write("Please provide the user input")