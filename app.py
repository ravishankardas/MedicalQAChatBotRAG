
import streamlit as st
import random
from constants import *
import tempfile
from get_graph import create_workflow

# Page config
st.set_page_config(
    page_title="🏥 Medical Assistant",
    page_icon="🏥",
    layout="centered"
)

# Title
st.title(st_title)
st.markdown(st_markdown)


workflow = create_workflow()


if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": st_welcome_message}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])



if prompt := st.chat_input("Ask a medical question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner(random.choice(spinner_messages)):

            try:
                result = workflow.invoke({
                    "question": prompt,
                    "answer": "",
                    "decision": ""
                })
                
                response = result['answer']
                
                route_used = result.get('decision', 'Unknown')
                if route_used == "EMERGENCY":
                    st.error(f"🚨 Emergency detected!")
                elif route_used == "RAG":
                    st.info("📚 Using medical knowledge base")
                else:
                    st.info("💬 General response")
                
                st.markdown(response)
                
            except Exception as e:
                error_msg = st_error_message
                st.error(error_msg)
                response = error_msg
                st.error(f"Error: {str(e)}")
    
    st.session_state.messages.append({"role": "assistant", "content": response})


with st.sidebar:
    st.header("ℹ️ Information")
    st.write(sidebar_messages)
    st.button("Clear Chat", on_click=lambda: st.session_state.pop("messages", None), help="Clear the chat history")

