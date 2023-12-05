import os
import streamlit as st
import pickle
import time
import pandas as pd
from io import StringIO
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

load_dotenv()
st.title("MyGen-AI:robot_face:")
st.sidebar.title("MyGen-AI:robot_face:")
st.sidebar.title("")
uploaded_file = st.sidebar.file_uploader("Choose a file", type=['txt'])

process_file_clicked = st.sidebar.button("Process File")
file_path = "vectorstore_openai.pkl"
main_placeholder = st.empty()
llm = OpenAI(temperature=0.9, max_tokens=100, model="gpt-3.5-turbo-instruct")
if process_file_clicked:
    # Save the uploaded file locally
    if uploaded_file is not None:
        with open("uploaded_file.txt", "wb") as f:
            f.write(uploaded_file.read())

    # Load a text file...................................................................................
    loader = TextLoader("uploaded_file.txt")
    main_placeholder.text("Loading file...")
    data = loader.load()
    text1 = data
    ab = text1[0].metadata
    st.sidebar.write(ab)
    
# Create a text splitter using recursive character splitter..............................................
    r_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " "],
        chunk_size=100,
        chunk_overlap=0
    )
    main_placeholder.text("Splitting text...")
    docs = r_splitter.split_documents(text1)

# create embeddings and save them to  FAISS index........................................................
    embeddings = OpenAIEmbeddings()
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Saving embeddings completed...")

# save the FAISS index to disk..........................................................................
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_openai, f)


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# asking a question.....................................................................................
query = st.chat_input("Say something")
    
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)

            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(query)
             # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": query})
            
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                assistant_response = result["answer"]
             # Simulate stream of response with milliseconds delay
                for chunk in assistant_response.split():
                   full_response += chunk + " "
                   time.sleep(0.05)
                   # Add a blinking cursor to simulate typing
                   message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})



            



