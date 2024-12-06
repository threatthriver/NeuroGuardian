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
        {"role": "assistant", "content": "Hello! I'm IntelliJMind AI, your intelligent companion. How can I assist you today?"}
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

# Set page configuration
st.set_page_config(
    page_title="IntelliJMind AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    st.title("🧠 Welcome to IntelliJMind AI")
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
            st.experimental_rerun()
    
    with col2:
        if st.button("Register", use_container_width=True):
            st.session_state.page = 'register'
            st.experimental_rerun()
    
    # About section
    st.write("---")
    st.header("ℹ️ About IntelliJMind AI")
    st.write("""
    IntelliJMind AI is an advanced chatbot powered by Cerebras AI technology. 
    It offers personalized conversations, multiple interaction modes, and a seamless user experience.
    Whether you need help with technical questions, creative writing, or academic research, 
    IntelliJMind AI is here to assist you.
    """)

# Login page
def show_login_page():
    st.title("🔐 Login to IntelliJMind AI")
    st.write("---")
    
    login_username = st.text_input("Username")
    login_password = st.text_input("Password", type="password")
    
    col1, col2, col3 = st.columns([1,1,1])
    
    with col1:
        if st.button("Back to Home"):
            st.session_state.page = 'landing'
            st.experimental_rerun()
    
    with col2:
        if st.button("Login"):
            if login_username in user_db and user_db[login_username]["password"] == hashlib.sha256(login_password.encode()).hexdigest():
                st.session_state.authenticated = True
                st.session_state.username = login_username
                st.success("Login successful!")
                time.sleep(1)
                st.experimental_rerun()
            else:
                st.error("Invalid credentials")
    
    with col3:
        if st.button("Need an account?"):
            st.session_state.page = 'register'
            st.experimental_rerun()

# Register page
def show_register_page():
    st.title("📝 Register for IntelliJMind AI")
    st.write("---")
    
    reg_username = st.text_input("Username")
    reg_password = st.text_input("Password", type="password")
    reg_email = st.text_input("Email")
    
    col1, col2, col3 = st.columns([1,1,1])
    
    with col1:
        if st.button("Back to Home"):
            st.session_state.page = 'landing'
            st.experimental_rerun()
    
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
                st.experimental_rerun()
    
    with col3:
        if st.button("Already have an account?"):
            st.session_state.page = 'login'
            st.experimental_rerun()

# Main application logic
if not st.session_state.authenticated:
    if st.session_state.page == 'landing':
        show_landing_page()
    elif st.session_state.page == 'login':
        show_login_page()
    elif st.session_state.page == 'register':
        show_register_page()
    st.stop()

# Update model name
MODEL_NAME = "IntelliJMind AI"

# Main chat interface
if st.session_state.authenticated:
    # Sidebar
    with st.sidebar:
        st.title(f"🎛️ Control Panel")
        st.sidebar.header(f"Welcome to {MODEL_NAME}")
        
        # User profile
        st.header("👤 User Profile")
        st.info(f"Logged in as: {st.session_state.username}")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Logout"):
                st.session_state.clear()
                st.experimental_rerun()
        with col2:
            if st.button("Home"):
                st.session_state.page = 'landing'
                st.session_state.authenticated = False
                st.experimental_rerun()
        
        # Model settings
        st.header("🤖 Model Settings")
        model = st.selectbox(
            "Model",
            ["llama3.1-70b", "llama3.1-8b"],
            help="Choose between powerful (70b) or fast (8b) model"
        )
        
        # Chat modes
        st.header("💭 Chat Mode")
        chat_mode = st.selectbox(
            "Mode",
            ["General", "Creative", "Technical", "Academic"],
            help="Select conversation style"
        )
        
        # Advanced settings
        with st.expander("⚙️ Advanced Settings"):
            temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
            max_tokens = st.slider("Max Tokens", 100, 500, 256)
            top_p = st.slider("Top P", 0.0, 1.0, 0.9)
        
        # Chat management
        st.header("💾 Chat Management")
        
        # Save chat
        chat_name = st.text_input("Save current chat as:")
        if st.button("Save Chat") and chat_name:
            if st.session_state.username not in chat_histories:
                chat_histories[st.session_state.username] = {}
            
            chat_histories[st.session_state.username][chat_name] = {
                "messages": st.session_state.messages,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            with open(CHAT_HISTORY_FILE, 'w') as f:
                json.dump(chat_histories, f)
            st.success(f"Chat saved as '{chat_name}'")
        
        # Load chat
        if st.session_state.username in chat_histories:
            saved_chats = list(chat_histories[st.session_state.username].keys())
            if saved_chats:
                selected_chat = st.selectbox("Load saved chat:", [""] + saved_chats)
                if selected_chat:
                    st.session_state.messages = chat_histories[st.session_state.username][selected_chat]["messages"]
                    st.success(f"Loaded chat: {selected_chat}")
        
        # Clear chat
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = [
                {"role": "assistant", "content": "Chat cleared. How can I help you?"}
            ]
            st.success("Chat history cleared!")
            st.experimental_rerun()
        
        # Chat analytics
        st.header("📊 Analytics")
        analytics = st.session_state.chat_analytics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Messages", analytics['total_messages'])
            st.metric("User Messages", analytics['user_messages'])
        with col2:
            st.metric("AI Responses", analytics['ai_messages'])
            if analytics['chat_duration']:
                avg_time = sum([d[1] for d in analytics['chat_duration']]) / len(analytics['chat_duration'])
                st.metric("Avg Response Time", f"{avg_time:.2f}s")
        
        if analytics['chat_duration']:
            df = pd.DataFrame(analytics['chat_duration'], columns=['timestamp', 'duration'])
            fig = px.line(df, x='timestamp', y='duration', title='Response Times')
            st.plotly_chart(fig, use_container_width=True)
        
        # User preferences for chat settings
        st.sidebar.header("User Preferences")
        if st.sidebar.button("Save Preferences"):
            st.session_state.user_preferences = {
                "model": MODEL_NAME,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p
            }
            st.success("Preferences saved!")
    
    # Main chat area
    st.title(f"🧠 {MODEL_NAME} Chat")
    st.write("---")
    
    # Function to handle AI responses with error handling
    def get_ai_response(prompt):
        try:
            client = get_cerebras_client()
            stream = client.chat.completions.create(
                messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
                model=model,
                stream=True,
                max_completion_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )
            return stream
        except Exception as e:
            st.error(f"An error occurred while fetching AI response: {str(e)}")
            if "rate limit" in str(e).lower():
                st.warning("You have reached the rate limit for the AI model. Please try again later.")
            return None

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Type your message..."):
        # Update analytics
        st.session_state.chat_analytics['total_messages'] += 1
        st.session_state.chat_analytics['user_messages'] += 1
        
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Get AI response with loading indicator
        with st.chat_message("assistant"):
            start_time = time.time()
            message_placeholder = st.empty()
            full_response = ""
            message_placeholder.markdown("Fetching response... Please wait.")
            
            # Call the AI response function
            stream = get_ai_response(prompt)
            if stream:
                for chunk in stream:
                    content = chunk.choices[0].delta.content or ""
                    full_response += content
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)

                # Update analytics
                response_time = time.time() - start_time
                st.session_state.chat_analytics['ai_messages'] += 1
                st.session_state.chat_analytics['chat_duration'].append([
                    datetime.now().strftime("%H:%M:%S"),
                    response_time
                ])

                # Add AI response to chat history
                st.session_state.messages.append({"role": "assistant", "content": full_response})

    # Function to view saved chat histories
    def view_saved_chats():
        if st.session_state.username in chat_histories:
            saved_chats = list(chat_histories[st.session_state.username].keys())
            if saved_chats:
                selected_chat = st.selectbox("Load saved chat:", [""] + saved_chats)
                if selected_chat:
                    st.session_state.messages = chat_histories[st.session_state.username][selected_chat]["messages"]
                    st.success(f"Loaded chat: {selected_chat}")
            else:
                st.info("No saved chats found.")
        else:
            st.info("No saved chats found for this user.")

    # Function to delete saved chat histories
    def delete_saved_chat(chat_name):
        if st.session_state.username in chat_histories:
            if chat_name in chat_histories[st.session_state.username]:
                del chat_histories[st.session_state.username][chat_name]
                with open(CHAT_HISTORY_FILE, 'w') as f:
                    json.dump(chat_histories, f)
                st.success(f"Deleted chat: {chat_name}")
            else:
                st.error("Chat not found.")
        else:
            st.error("No saved chats found for this user.")

    # Styling enhancements
    st.markdown("<style>.stChatMessage { border-radius: 10px; padding: 10px; }</style>", unsafe_allow_html=True)
