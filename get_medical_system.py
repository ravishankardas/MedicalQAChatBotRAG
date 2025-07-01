from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_openai import ChatOpenAI


from sentence_transformers import CrossEncoder
from dotenv import load_dotenv
import streamlit as st
from datasets import load_dataset
import os

load_dotenv()

    
@st.cache_resource
def load_medical_system():
    """Load the medical RAG system (cached for performance)"""
    
    with st.spinner("🔄 Loading medical knowledge base..."):
        # Load dataset
        ds = load_dataset("keivalya/MedQuad-MedicalQnADataset")
        
        # Create documents
        documents = []
        for i, item in enumerate(ds['train']): # type: ignore
            content = f"Question: {item['Question']}\\nAnswer: {item['Answer']}" # type: ignore
            metadata = {
                "doc_id": i,
                "question": item['Question'], # type: ignore
                "answer": item['Answer'], # type: ignore
                "question_type": item['qtype'], # type: ignore
                "type": "qa_pair"
            }
            documents.append(Document(page_content=content, metadata=metadata))
        
        # Create embeddings and vectorstore
        embeddings = HuggingFaceEmbeddings(model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
        
        # Try to load existing vectorstore, create if doesn't exist
        try:
            vectorstore = Chroma(persist_directory="medical_vectordb", embedding_function=embeddings)
            st.success("✅ Loaded existing vectorstore")
        except:
            st.info("📦 Creating new vectorstore...")
            vectorstore = Chroma.from_documents(documents, embeddings, persist_directory="medical_vectordb")
            st.success("✅ Created new vectorstore")
        
        # Create retrievers
        bm25_retriever = BM25Retriever.from_documents(documents)
        vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
        
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.3, 0.7]
        )
        
        # create LLM
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            st.error("❌ OpenAI API key not found! Please set it in your environment variables or .streamlit/secrets.toml")
            st.stop()
        llm = ChatOpenAI(temperature=0, max_tokens=512, api_key=openai_key) # type: ignore
        
        # Create reranker
        reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        return documents, ensemble_retriever, llm, reranker
