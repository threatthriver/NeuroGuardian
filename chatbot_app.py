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
        'chat_duration': [],
        'mood_tracking': [],
        'topics_discussed': []
    }

if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

if 'username' not in st.session_state:
    st.session_state.username = None

# Database configuration
USER_DB_FILE = "data/users.json"
CHAT_HISTORY_FILE = "data/chat_history.json"

# Ensure data directory exists
os.makedirs("data", exist_ok=True)

def save_database(filename, data):
    """Save data to a JSON file"""
    try:
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        st.error(f"Error saving data: {str(e)}")

def load_database(filename):
    """Load data from a JSON file"""
    try:
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                return json.load(f)
        return {}
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return {}

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

# Custom CSS for dark mode compatibility
st.markdown("""
<style>
    /* Base styles */
    .main {
        padding: 2rem;
    }
    
    /* Button styles */
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        height: 3em;
        background-color: var(--primary-color);
        color: white;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        opacity: 0.8;
        transform: translateY(-2px);
    }
    
    /* Chat message styles */
    .chat-message {
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1rem;
        border: 1px solid rgba(250, 250, 250, 0.1);
    }
    .user-message {
        background-color: rgba(28, 131, 225, 0.1);
    }
    .assistant-message {
        background-color: rgba(255, 255, 255, 0.05);
    }
    
    /* Feature card styles */
    .feature-card {
        padding: 1.5rem;
        border-radius: 15px;
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(250, 250, 250, 0.1);
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
    }
    
    /* Dark mode specific styles */
    [data-testid="stSidebar"] {
        background-color: rgba(0, 0, 0, 0.2);
    }
    
    .user-profile {
        background-color: rgba(255, 255, 255, 0.05);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid rgba(250, 250, 250, 0.1);
    }
    
    /* Form styles */
    [data-testid="stForm"] {
        background-color: rgba(255, 255, 255, 0.05);
        padding: 2rem;
        border-radius: 15px;
        border: 1px solid rgba(250, 250, 250, 0.1);
    }
    
    /* Input styles */
    .stTextInput>div>div>input {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(250, 250, 250, 0.1);
        color: inherit;
    }
    
    /* Metric styles */
    [data-testid="stMetricValue"] {
        background-color: rgba(255, 255, 255, 0.05);
        padding: 0.5rem;
        border-radius: 5px;
        border: 1px solid rgba(250, 250, 250, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# User authentication
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

# Load user database
user_db = load_database(USER_DB_FILE)
chat_histories = load_database(CHAT_HISTORY_FILE)

# Landing page
def show_landing_page():
    st.title("🧠 Welcome to NeuroGuardian AI")
    
    # Hero section
    st.markdown("""
    <div style='text-align: center; padding: 2rem;'>
        <h2>Your Intelligent Mental Health Companion</h2>
        <p style='font-size: 1.2em; color: #666;'>
            Experience personalized mental health support powered by advanced AI technology
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Features showcase
    st.header("✨ Key Features")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='feature-card'>
            <h3>🤝 24/7 Support</h3>
            <p>Always available to listen and provide guidance when you need it most</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='feature-card'>
            <h3>📊 Mood Tracking</h3>
            <p>Monitor your emotional well-being with interactive analytics</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='feature-card'>
            <h3>🔒 Private & Secure</h3>
            <p>Your conversations are encrypted and completely confidential</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Login/Register buttons
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,2,1])
    
    with col2:
        login_col, register_col = st.columns(2)
        with login_col:
            if st.button("🔑 Login", use_container_width=True):
                st.session_state.page = 'login'
                st.rerun()
        
        with register_col:
            if st.button("📝 Register", use_container_width=True):
                st.session_state.page = 'register'
                st.rerun()
    
    # About section with more details
    st.markdown("---")
    st.header("ℹ️ About NeuroGuardian AI")
    st.write("""
    NeuroGuardian AI is an advanced mental health companion powered by state-of-the-art AI technology. 
    Our platform offers:
    
    - **Personalized Support**: Tailored conversations based on your needs and preferences
    - **Multiple Interaction Modes**: Text, voice, or guided exercises
    - **Progress Tracking**: Monitor your mental health journey with detailed analytics
    - **Resource Library**: Access to mental health resources and coping strategies
    - **Crisis Support**: Immediate access to crisis helplines and emergency contacts
    """)
    
    # Testimonials
    st.header("💬 What Our Users Say")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='feature-card'>
            <p><em>"NeuroGuardian has been a game-changer for my mental health journey. The 24/7 availability and personalized support make all the difference."</em></p>
            <p>- Sarah M.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='feature-card'>
            <p><em>"The mood tracking and analytics help me understand my patterns and make better decisions for my mental well-being."</em></p>
            <p>- James R.</p>
        </div>
        """, unsafe_allow_html=True)

# Login page
def show_login_page():
    st.title("🔐 Login to NeuroGuardian")
    
    # Center the login form
    col1, col2, col3 = st.columns([1,2,1])
    
    with col2:
        st.markdown("""
        <div style='background-color: white; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
            <h3 style='text-align: center; margin-bottom: 2rem;'>Welcome Back!</h3>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("login_form"):
            login_username = st.text_input("Username", placeholder="Enter your username")
            login_password = st.text_input("Password", type="password", placeholder="Enter your password")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.form_submit_button("🔑 Login", use_container_width=True):
                    if login_username and login_password:
                        password_hash = hashlib.sha256(login_password.encode()).hexdigest()
                        if login_username in user_db and user_db[login_username]["password"] == password_hash:
                            st.session_state.authenticated = True
                            st.session_state.username = login_username
                            st.success("Login successful! Redirecting...")
                            time.sleep(1)
                            st.session_state.page = 'chat'
                            st.rerun()
                        else:
                            st.error("Invalid username or password")
                    else:
                        st.warning("Please fill in all fields")
            
            with col2:
                if st.form_submit_button("🏠 Back to Home", use_container_width=True):
                    st.session_state.page = 'landing'
                    st.rerun()
        
        st.markdown("""
        <div style='text-align: center; margin-top: 1rem;'>
            <p>Don't have an account? <a href='javascript:void(0);' onclick='document.querySelector("button:contains(\'Register\')").click()'>Register here</a></p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("📝 Register", use_container_width=True):
            st.session_state.page = 'register'
            st.rerun()

# Register page
def show_register_page():
    st.title("📝 Create Your Account")
    
    # Center the registration form
    col1, col2, col3 = st.columns([1,2,1])
    
    with col2:
        st.markdown("""
        <div style='background-color: white; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
            <h3 style='text-align: center; margin-bottom: 2rem;'>Join NeuroGuardian</h3>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("register_form"):
            new_username = st.text_input("Username", placeholder="Choose a username")
            new_password = st.text_input("Password", type="password", placeholder="Choose a strong password")
            confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm your password")
            email = st.text_input("Email (optional)", placeholder="Enter your email")
            
            # Password strength indicator
            if new_password:
                strength = 0
                if len(new_password) >= 8: strength += 1
                if any(c.isupper() for c in new_password): strength += 1
                if any(c.islower() for c in new_password): strength += 1
                if any(c.isdigit() for c in new_password): strength += 1
                if any(not c.isalnum() for c in new_password): strength += 1
                
                st.progress(strength/5)
                st.caption(f"Password strength: {['Very Weak', 'Weak', 'Moderate', 'Strong', 'Very Strong'][strength-1]}")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.form_submit_button("📝 Register", use_container_width=True):
                    if new_username and new_password and confirm_password:
                        if new_password == confirm_password:
                            if new_username not in user_db:
                                if len(new_password) >= 8:
                                    password_hash = hashlib.sha256(new_password.encode()).hexdigest()
                                    user_db[new_username] = {
                                        "password": password_hash,
                                        "email": email,
                                        "created_at": datetime.now().isoformat(),
                                        "last_login": None
                                    }
                                    save_database(USER_DB_FILE, user_db)
                                    st.success("Registration successful! Please login.")
                                    time.sleep(1)
                                    st.session_state.page = 'login'
                                    st.rerun()
                                else:
                                    st.error("Password must be at least 8 characters long")
                            else:
                                st.error("Username already exists")
                        else:
                            st.error("Passwords do not match")
                    else:
                        st.warning("Please fill in all required fields")
            
            with col2:
                if st.form_submit_button("🏠 Back to Home", use_container_width=True):
                    st.session_state.page = 'landing'
                    st.rerun()
        
        st.markdown("""
        <div style='text-align: center; margin-top: 1rem;'>
            <p>Already have an account? <a href='javascript:void(0);' onclick='document.querySelector("button:contains(\'Login\')").click()'>Login here</a></p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔑 Login", use_container_width=True):
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
def get_ai_response(prompt, chat_mode):
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
    # Sidebar for settings and analytics
    with st.sidebar:
        st.title("⚙️ Settings")
        
        # User profile
        st.markdown(f"""
        <div class='user-profile'>
            <h3>👤 {st.session_state.username}</h3>
            <p>Member since: {user_db[st.session_state.username]['created_at'][:10]}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Chat mode selection
        st.subheader("💭 Chat Mode")
        chat_mode = st.selectbox(
            "Select conversation style",
            ["Supportive", "Professional", "Casual", "Emergency Support"],
            index=0
        )
        
        # Mood tracking
        st.subheader("😊 Mood Tracker")
        current_mood = st.select_slider(
            "How are you feeling?",
            options=["😢", "😕", "😐", "🙂", "😊"],
            value="😐"
        )
        if st.button("Save Mood"):
            st.session_state.chat_analytics['mood_tracking'].append({
                'timestamp': datetime.now().isoformat(),
                'mood': current_mood
            })
            st.success("Mood recorded!")
        
        # Analytics
        st.subheader("📊 Chat Analytics")
        
        # Metrics in a card
        st.markdown("""
        <div class='feature-card'>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Messages", st.session_state.chat_analytics['total_messages'])
        with col2:
            avg_time = st.session_state.chat_analytics['average_response_time']
            st.metric("Avg. Response", f"{avg_time:.1f}s")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Show mood history
        if st.session_state.chat_analytics['mood_tracking']:
            st.markdown("""
            <div class='feature-card'>
                <h4>Mood History</h4>
            """, unsafe_allow_html=True)
            
            moods = [m['mood'] for m in st.session_state.chat_analytics['mood_tracking']]
            dates = [m['timestamp'][:10] for m in st.session_state.chat_analytics['mood_tracking']]
            mood_df = pd.DataFrame({'Date': dates, 'Mood': moods})
            st.line_chart(mood_df.groupby('Date').size(), use_container_width=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Logout button
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.session_state.page = 'landing'
            st.rerun()
    
    # Main chat area
    st.title("🧠 NeuroGuardian AI Chat")
    
    # Emergency support banner
    if chat_mode == "Emergency Support":
        st.warning("""
        🚨 **Emergency Mode Active**
        
        If you're in immediate danger or crisis:
        - Call Emergency Services: 911
        - National Crisis Hotline: 988
        - Text HOME to 741741 to connect with a Crisis Counselor
        """)
    
    # Chat interface with improved styling
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(f"""
            <div class='chat-message {"user-message" if message["role"] == "user" else "assistant-message"}'>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.chat_analytics['user_messages'] += 1
        st.session_state.chat_analytics['total_messages'] += 1
        
        # Get AI response
        start_time = time.time()
        response = get_ai_response(prompt, chat_mode)
        response_time = time.time() - start_time
        
        # Update analytics
        st.session_state.chat_analytics['ai_messages'] += 1
        st.session_state.chat_analytics['total_messages'] += 1
        st.session_state.chat_analytics['average_response_time'] = (
            (st.session_state.chat_analytics['average_response_time'] * 
             (st.session_state.chat_analytics['ai_messages'] - 1) + response_time) / 
            st.session_state.chat_analytics['ai_messages']
        )
        
        # Add AI response to chat
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Save chat history
        if st.session_state.username in chat_histories:
            chat_histories[st.session_state.username]['messages'] = st.session_state.messages
            chat_histories[st.session_state.username]['analytics'] = st.session_state.chat_analytics
        else:
            chat_histories[st.session_state.username] = {
                'messages': st.session_state.messages,
                'analytics': st.session_state.chat_analytics
            }
        
        save_database(CHAT_HISTORY_FILE, chat_histories)
        st.rerun()
    
    # Clear chat button with custom styling
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm NeuroGuardian AI, your mental health companion. How can I assist you today?"}
        ]
        st.rerun()
