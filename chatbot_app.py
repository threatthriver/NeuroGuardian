import streamlit as st
import os
from datetime import datetime
from cerebras.cloud.sdk import Cerebras
from dotenv import load_dotenv
import hashlib
import json
import time
import pandas as pd
import plotly.express as px
from pathlib import Path

# Load environment variables
load_dotenv()

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'landing'

if 'messages' not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm NeuroGuardian AI, your mental health companion. How can I assist you today?"}
    ]

if 'chat_analytics' not in st.session_state:
    st.session_state.chat_analytics = {
        'total_messages': 0,
        'user_messages': 0,
        'ai_messages': 0,
        'average_response_time': 0,
        'chat_duration': []
    }

if 'show_register' not in st.session_state:
    st.session_state.show_register = False

# Initialize Cerebras client
@st.cache_resource
def get_cerebras_client():
    return Cerebras(api_key=os.environ.get("CEREBRAS_API_KEY"))

# Page configuration for better UI
st.set_page_config(
    page_title="NeuroGuardian AI Chat",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/NeuroGuardian',
        'Report a bug': "https://github.com/yourusername/NeuroGuardian/issues",
        'About': "# NeuroGuardian AI Chat\nAn intelligent AI companion for mental health support."
    }
)

# Add custom CSS for better styling
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .assistant {
        background-color: #f0f2f6;
    }
    .user {
        background-color: #e6f3ff;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

# User authentication
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

# Load user database
USER_DB_FILE = "user_db.json"
CHAT_HISTORY_FILE = "chat_histories.json"

def load_database(filename):
    if not os.path.exists(filename):
        with open(filename, 'w') as f:
            json.dump({}, f)
    with open(filename, 'r') as f:
        return json.load(f)

user_db = load_database(USER_DB_FILE)
chat_histories = load_database(CHAT_HISTORY_FILE)

# Landing page
def show_landing_page():
    st.title("🧠 Welcome to NeuroGuardian AI")
    st.subheader("Your Intelligent AI Companion")
    
    # Features section
    st.write("---")
    st.header("✨ Key Features")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🤖 Multiple AI Modes")
        st.write("Choose from General, Creative, Technical, and Academic conversation styles.")
    
    with col2:
        st.markdown("### 💾 Chat History")
        st.write("Save and load your conversations anytime.")
    
    with col3:
        st.markdown("### 📊 Analytics")
        st.write("Track your interaction patterns and AI response times.")
    
    # Call to action
    st.write("---")
    st.header("🚀 Get Started")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Login", use_container_width=True):
            st.session_state.page = 'login'
            st.rerun()
    
    with col2:
        if st.button("Register", use_container_width=True):
            st.session_state.page = 'register'
            st.rerun()
    
    # About section
    st.write("---")
    st.header("ℹ️ About NeuroGuardian AI")
    st.write("""
    NeuroGuardian AI is an advanced chatbot powered by Cerebras AI technology. 
    It offers personalized conversations, multiple interaction modes, and a seamless user experience.
    Whether you need help with technical questions, creative writing, or academic research, 
    NeuroGuardian AI is here to assist you.
    """)

# Login page
def show_login_page():
    st.title("🔐 Login to NeuroGuardian AI")
    st.write("---")
    
    login_username = st.text_input("Username")
    login_password = st.text_input("Password", type="password")
    
    col1, col2, col3 = st.columns([1,1,1])
    
    with col1:
        if st.button("Back to Home"):
            st.session_state.page = 'landing'
            st.rerun()
    
    with col2:
        if st.button("Login"):
            if login_username in user_db and user_db[login_username]["password"] == hashlib.sha256(login_password.encode()).hexdigest():
                st.session_state.authenticated = True
                st.session_state.username = login_username
                st.success("Login successful!")
                time.sleep(1)
                st.rerun()
            else:
                st.error("Invalid credentials")
    
    with col3:
        if st.button("Need an account?"):
            st.session_state.page = 'register'
            st.rerun()

# Register page
def show_register_page():
    st.title("📝 Register for NeuroGuardian AI")
    st.write("---")
    
    reg_username = st.text_input("Username")
    reg_password = st.text_input("Password", type="password")
    reg_email = st.text_input("Email")
    
    col1, col2, col3 = st.columns([1,1,1])
    
    with col1:
        if st.button("Back to Home"):
            st.session_state.page = 'landing'
            st.rerun()
    
    with col2:
        if st.button("Register"):
            if reg_username in user_db:
                st.error("Username already exists")
            else:
                user_db[reg_username] = {
                    "password": hashlib.sha256(reg_password.encode()).hexdigest(),
                    "email": reg_email,
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                with open(USER_DB_FILE, 'w') as f:
                    json.dump(user_db, f)
                st.success("Registration successful!")
                time.sleep(1)
                st.session_state.page = 'login'
                st.rerun()
    
    with col3:
        if st.button("Already have an account?"):
            st.session_state.page = 'login'
            st.rerun()

# Main application logic
if not st.session_state.authenticated:
    if st.session_state.page == 'landing':
        show_landing_page()
    elif st.session_state.page == 'login':
        show_login_page()
    elif st.session_state.page == 'register':
        show_register_page()
    st.stop()

# Function to handle AI responses with improved error handling and streaming
def get_ai_response(prompt):
    try:
        with st.spinner('🤔 Thinking...'):
            client = get_cerebras_client()
            
            # Add system message for context
            messages = [
                {"role": "system", "content": "You are NeuroGuardian, an AI mental health companion. Respond with empathy and professionalism."},
                *[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
            ]
            
            stream = client.chat.completions.create(
                messages=messages,
                model="llama3.1-70b",
                stream=True,
                max_completion_tokens=500,
                temperature=0.7,
                top_p=0.9
            )
            return stream
    except Exception as e:
        error_message = str(e).lower()
        if "rate limit" in error_message:
            st.error("⚠️ Rate limit reached. Please try again in a few moments.")
        elif "connection" in error_message:
            st.error("📶 Connection error. Please check your internet connection.")
        else:
            st.error(f"🚫 An error occurred: {str(e)}")
        return None

# Main chat interface with improved UI
if st.session_state.authenticated:
    st.title("🧠 NeuroGuardian AI Chat")
    st.markdown("---")
    
    # Add sidebar for settings
    with st.sidebar:
        st.subheader("Chat Settings")
        temperature = st.slider("Creativity", 0.0, 1.0, 0.7)
        max_tokens = st.slider("Max Response Length", 100, 1000, 500)
        
        if st.button("Clear Chat"):
            st.session_state.messages = [
                {"role": "assistant", "content": "Hello! I'm NeuroGuardian AI, your mental health companion. How can I assist you today?"}
            ]
            st.rerun()
    
    # Chat interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            stream = get_ai_response(prompt)
            if stream:
                for chunk in stream:
                    content = chunk.choices[0].delta.content or ""
                    full_response += content
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
                
                # Update analytics
                st.session_state.chat_analytics['total_messages'] += 1
                st.session_state.chat_analytics['ai_messages'] += 1
                
                # Add to chat history
                st.session_state.messages.append({"role": "assistant", "content": full_response})
