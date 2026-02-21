import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


#..................................................................
#page config
#..................................................................
st.set_page_config(page_title="C++ RAG Chatbot")
st.title("C++ RAG Chatbot")
st.write("Ask any question related to C++ Introduction")


#..................................................................
#Load Environment Variables
#..................................................................
#load_dotenv()


#..................................................................
#Cache Document loading
#..................................................................

@st.cache_resource
def load_vectorstore():
    #load Document
    loader=TextLoader('C++_Introduction.txt',encoding="utf-8")
    documents=loader.load()

    #split text
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )
    final_documents=text_splitter.split_documents(documents)


    #Embeddings
    embeddings=HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    #create FAISS Vector store
    db=FAISS.from_documents(final_documents,embeddings)
    return db


#load Vector DB(only once)
db=load_vectorstore()


#..................................................................
#User Input
#..................................................................

query=st.text_input("Enter your Question for C++")
if query:
    docs=db.similarity_search(query,k=3)
    st.subheader("Retrived Context:")
    for i,doc in enumerate(docs):
       st.markdown(f"**Result {i+1}:**")
       st.write(docs[i].page_content)



   