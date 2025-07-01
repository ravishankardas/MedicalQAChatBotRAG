from typing import TypedDict, Literal
from pydantic import Field
from pydantic import BaseModel
import streamlit as st
from langchain_core.messages import HumanMessage, SystemMessage
from get_medical_system import load_medical_system
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import AmazonTextractPDFLoader
from langgraph.graph import StateGraph, END, START
from langchain_community.document_loaders import UnstructuredPDFLoader


class Route(BaseModel):
    step: Literal["RAG", "GENERAL", "EMERGENCY", "MEMORY"] = Field(None, description="The next step in the routing process") # type: ignore

class State(TypedDict):
    question: str
    answer: str
    decision: str


def init_document_memory():
    """Initialize document memory in session state"""
    if "uploaded_documents" not in st.session_state:
        st.session_state.uploaded_documents = {}

documents, ensemble_retriever, llm, reranker = load_medical_system()
router = llm.with_structured_output(Route, method="function_calling")

def extract_conversation_history():
    """Extract conversation from session state"""
    if "messages" not in st.session_state:
        return []
    
    conversation = []
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            conversation.append(f"User: {msg['content']}")
        elif msg["role"] == "assistant" and not msg["content"].startswith("Hello!"):
            conversation.append(f"Assistant: {msg['content']}")
    
    return conversation

def handle_conversation_query(state: State):
    """Handle questions about conversation history"""
    
    conversation = extract_conversation_history()
    
    if not conversation:
        return {"answer": "We haven't had any conversation yet. Feel free to ask me a medical question though!"}
    
    # Create conversation context
    conversation_text = "\n".join(conversation[-10:])  # Last 10 exchanges
    
    result = llm.invoke([
        SystemMessage(content=f"""
Based on this conversation history, answer the user's question about our previous discussion:

Conversation History:
{conversation_text}

Rules:
- If they ask for a summary, provide a brief overview
- If they ask about specific questions, reference them
- If they ask about previous answers, summarize the key points
- Always maintain medical disclaimers in your response
        """),
        HumanMessage(content=state['question'])
    ])
    
    return {"answer": result.content}

def is_conversation_query(question: str) -> bool:
    """Check if the question is about conversation history"""
    memory_keywords = [
        "previous", "last", "earlier", "before", "summarize", "summarise", 
        "what did i ask", "my questions", "conversation", "history", 
        "we talked", "discussed", "mentioned"
    ]
    
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in memory_keywords)


def llm_call_router(state: State):
    """Enhanced router that includes document routing"""
    # if st.session_state.get("current_document"):
    #     return {'decision': "DOCUMENT"}
    
    # Check for conversation/memory queries FIRST
    if is_conversation_query(state['question']):
        return {'decision': "MEMORY"}
    
    # Check if question is about an uploaded document
    # document_keywords = ["document", "report", "lab results", "test results", "my results", "uploaded", "file"]
    # if any(keyword in state['question'].lower() for keyword in document_keywords):
    #     if "current_document" in st.session_state and st.session_state.current_document:
    #         return {'decision': "DOCUMENT"}
    
    # Emergency check
    emergency_keywords = ["severe", "chest pain", "can't breathe", "emergency", "urgent",
                         "heart attack", "stroke", "bleeding", "unconscious"]
    question_lower = state['question'].lower()
    if any(keyword in question_lower for keyword in emergency_keywords):
        return {'decision': "EMERGENCY"}

    # Regular routing
    decision = router.invoke([
        SystemMessage(content="Route the input to RAG (medical questions) or GENERAL based on the user's request"),
        HumanMessage(content=state['question'])
    ])
    return {"decision": decision.step} # type: ignore

def emergency_node(state: State):
    """Handle emergency queries safely"""
    return {"answer": "🚨 EMERGENCY: Please seek immediate medical attention or call emergency services (911). This system cannot provide emergency medical care."}

def rag_node(state: State):
    """Uses RAG to answer the question with reranking"""
    
    custom_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a medical information assistant. Use the following medical Q&A context to answer questions accurately and safely.

        Context: {context}

        Question: {question}

        Guidelines:
        - Provide accurate medical information based on the context above
        - Always recommend consulting healthcare professionals for medical decisions
        - If uncertain, clearly state limitations
        - If the question is not suitable for this bot, respond with: "I'm not able to provide medical advice. Please consult a medical professional."

        Answer:"""
    )
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=ensemble_retriever,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": custom_prompt}
    )

    result = qa_chain.invoke({
        "question": state['question'],
        "chat_history": []
    })

    # Reranking
    docs = result.get('source_documents', [])
    if docs and len(docs) > 1:
        pairs = [(state['question'], doc.page_content) for doc in docs]
        scores = reranker.predict(pairs)

        doc_scores = list(zip(docs, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        top_docs = [doc for doc, score in doc_scores[:3]]

        better_context = "\\n\\n".join([doc.page_content for doc in top_docs])
        improved_answer = llm.invoke([
            SystemMessage(content=f"""Use this medical context to answer the question safely:

            Context: {better_context}

            Always recommend consulting healthcare professionals."""),
            HumanMessage(content=state['question'])
        ])
        return {"answer": improved_answer.content}

    return {"answer": result['answer']}

def general_node(state: State):
    """Enhanced general node with sarcastic responses for identity questions"""
    
    question_lower = state['question'].lower().strip()
    
    # Identity/philosophical questions - sarcastic responses
    identity_keywords = [
        "what are you", "who are you", "what is your name", "are you human", 
        "are you real", "are you ai", "are you robot", "are you chatbot",
        "what's your name", "who made you", "are you alive", "do you think",
        "are you conscious", "do you feel", "what do you do", "your purpose"
    ]
    
    if any(keyword in question_lower for keyword in identity_keywords):
        # Sarcastic responses for identity questions
        sarcastic_responses = [
            "🤖 Oh, just your friendly neighborhood medical AI trying to keep people from WebMD-ing themselves into thinking they have every disease known to humanity. You know, the usual.",
            
            "🩺 I'm a sophisticated medical assistant, which is a fancy way of saying I'm here to tell you to 'consult a healthcare professional' in 47 different ways.",
            
            "🏥 I'm an AI that reads medical textbooks faster than you can say 'Google symptoms at 3 AM.' My purpose? Giving you actual medical info instead of letting you convince yourself that headache is definitely a brain tumor.",
            
            "💊 I'm basically a walking medical disclaimer with a personality. Think of me as that friend who went to med school but actually remembers what they learned.",
            
            "🔬 I'm an artificial intelligence trained on medical knowledge, which means I can tell you about symptoms but I still can't fix your tendency to ignore doctor's appointments.",
            
            "🧠 I'm a medical AI assistant. I exist to answer your health questions and remind you that, no, that WebMD article probably doesn't apply to you."
        ]
        
        import random
        return {"answer": random.choice(sarcastic_responses)}
    
    # Greeting responses - also with some personality
    greeting_keywords = ["hello", "hi", "hey", "good morning", "good evening", "greetings"]
    if any(keyword in question_lower for keyword in greeting_keywords):
        friendly_responses = [
            "Hello! 👋 Ready to get some actual medical information instead of falling down a WebMD rabbit hole?",
            "Hi there! 🏥 I'm here to answer your medical questions. Fair warning: I'll probably tell you to see a real doctor.",
            "Hey! 👨‍⚕️ What medical mystery can I help solve today? (Spoiler: the answer might be 'drink more water')",
            "Greetings! 🩺 Ask me anything medical-related. I promise to give you better advice than your cousin's Facebook post."
        ]
        
        import random
        return {"answer": random.choice(friendly_responses)}
    
    # Regular medical or general questions
    result = llm.invoke([
        SystemMessage(content="""
Answer the user's question helpfully and accurately.

IMPORTANT SAFETY RULES:
- For medical questions: Always end with "Please consult a healthcare professional"
- For emergencies: Direct to call emergency services immediately  
- If unsure: Say "I don't know" rather than guess

Be helpful but prioritize user safety. You can be slightly witty or conversational, but always maintain professionalism for serious medical topics.
        """),
        HumanMessage(content=state['question'])
    ])

    return {"answer": result.content}

def document_node(state: State):
    """Simple document processing node that integrates with your existing workflow"""
    
    
    # Check if there's an uploaded document in session state
    if "current_document" not in st.session_state or not st.session_state.current_document:
        return {"answer": "Please upload a medical document first using the file uploader in the sidebar."}
    
    file_path = st.session_state.current_document
    question = state['question']
    
    try:
        # Check if document already processed
        if file_path not in st.session_state.uploaded_documents:
            # Extract document content
            # loader = AmazonTextractPDFLoader(file_path, region_name="us-east-1")
            loader = UnstructuredPDFLoader(file_path)
            documents = loader.load()
            
            # Clean and store content
            content = "\n".join([doc.page_content for doc in documents])
            st.session_state.uploaded_documents[file_path] = {
                "content": content,
                "conversation": []
            }
        
        # Get stored document
        doc_data = st.session_state.uploaded_documents[file_path]
        
        # Build context with previous questions about this document
        context_parts = [f"Document Content:\n{doc_data['content']}"]
        
        if doc_data['conversation']:
            context_parts.append("\nPrevious questions about this document:")
            for qa in doc_data['conversation'][-3:]:  # Last 3 Q&As
                context_parts.append(f"Q: {qa['question']}\nA: {qa['answer'][:200]}...")
        
        full_context = "\n".join(context_parts)
        
        # Generate answer using your existing LLM
        from langchain_core.messages import HumanMessage, SystemMessage
        
        result = llm.invoke([
            SystemMessage(content=f"""
            You are analyzing a medical document. Use the document content and any previous conversation to answer the user's question.
            
            Guidelines:
            - Base your answer on the document content provided
            - Reference specific values or sections when possible
            - If information isn't in the document, clearly state this
            - Always include medical disclaimers
            - Maintain conversation continuity with previous questions
            
            {full_context}
            """),
            HumanMessage(content=f"Question about the document: {question}")
        ])
        
        # Store this Q&A in document conversation history
        doc_data['conversation'].append({
            "question": question,
            "answer": result.content
        })
        
        return {"answer": f"📄 **Document Analysis:**\n\n{result.content}"}
        
    except Exception as e:
        return {"answer": f"Error processing document: {str(e)}. Please ensure the file is accessible and try again."}


def route_decision(state: State):
    """Enhanced route decision with memory"""
    if state["decision"] == "MEMORY":
        return "memory_node"
    elif state["decision"] == "DOCUMENT":
        return "document_node"
    elif state["decision"] == "RAG":
        return "rag_node"
    elif state["decision"] == "EMERGENCY":
        return "emergency_node"
    else:
        return "general_node"

# ==================== CREATE WORKFLOW ====================

@st.cache_resource
def create_workflow():
    """Create the enhanced workflow graph with memory"""
    
    init_document_memory()

    router_builder = StateGraph(State)
    
    # Add all nodes (including new memory node)
    router_builder.add_node("rag_node", rag_node)
    router_builder.add_node("general_node", general_node)
    router_builder.add_node("llm_call_router", llm_call_router)
    router_builder.add_node("emergency_node", emergency_node)
    router_builder.add_node("memory_node", handle_conversation_query)  # NEW NODE
    # router_builder.add_node("document_node", document_node)

    
    router_builder.add_edge(START, "llm_call_router")
    router_builder.add_conditional_edges(
        "llm_call_router",
        route_decision,
        {
            "rag_node": "rag_node",
            "general_node": "general_node",
            "emergency_node": "emergency_node",
            "memory_node": "memory_node",  # NEW ROUTE,
            # "document_node": "document_node" 
        },
    )
    
    # Add edges to END
    router_builder.add_edge("rag_node", END)
    router_builder.add_edge("general_node", END)
    router_builder.add_edge("emergency_node", END)
    router_builder.add_edge("memory_node", END)  # NEW EDGE
    # router_builder.add_edge("document_node", END)

    return router_builder.compile()


