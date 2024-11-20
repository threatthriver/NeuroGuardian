import os
import uuid
import streamlit as st
from groq import Groq, RateLimitError, APIError
from dotenv import load_dotenv
import pandas as pd
import requests

# Load environment variables
load_dotenv()

# Configuration and Setup
st.set_page_config(
    page_title="MediAssist AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS with Dark Mode Support and Improved Color Palette
st.markdown("""
<style>
    body {
        background-color: #1e212d; /* Sophisticated dark background */
        color: #dfe6e9; /* Light gray text for readability */
        font-family: 'Inter', sans-serif;
        transition: background-color 0.3s ease, color 0.3s ease;
    }

    .main-header {
        background: linear-gradient(135deg, #2a3042 0%, #3e4558 100%); /* Subtle gradient for header */
        color: #dfe6e9;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    .sidebar .sidebar-content {
        background-color: #2a3042; /* Darker sidebar for contrast */
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        color: #dfe6e9;
    }

    .stContainer {
        background-color: #2a3042; /* Darker container for content */
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        color: #dfe6e9;
    }

    .stButton>button {
        background-color: #4285f4; /* Google blue for buttons */
        color: white;
        border: none;
        border-radius: 6px;
        padding: 10px 20px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #3273dc; /* Darker blue on hover */
        transform: translateY(-2px);
    }

    .user-message {
        background-color: #3e4558; /* User message background */
        border-left: 4px solid #4285f4; /* Google blue border */
        padding: 10px;
        margin-bottom: 10px;
        border-radius: 5px;
        color: #dfe6e9;
    }
    .ai-message {
        background-color: #2a3042; /* AI message background */
        border-left: 4px solid #a5d6a7; /* Green border for AI messages */
        padding: 10px;
        margin-bottom: 10px;
        border-radius: 5px;
        color: #dfe6e9;
    }

    /* Media query for dark mode */
    @media (prefers-color-scheme: dark) {
        body {
            background-color: #1e212d; /* Consistent dark background */
            color: #dfe6e9; /* Consistent light text */
        }
    }
</style>
""", unsafe_allow_html=True)

class MedicalAIChatbot:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            st.error("API key not found. Please set GROQ_API_KEY in your .env file.")
            raise EnvironmentError("API key not found.")
        self.client = Groq(api_key=api_key)
        self.system_prompt = """You are MediAssist, an advanced AI medical companion designed to:
        - Provide comprehensive, evidence-based medical insights
        - Support healthcare professionals with clinical reasoning
        - Explain medical concepts clearly and precisely

        Key Principles:
        1. Clarify that you are an AI assistant, not a substitute for professional medical advice
        2. Use clear and empathetic language
        3. Simplify complex medical information
        4. Prioritize patient safety and understanding
        5. Recommend professional consultation when necessary

        Communication Style:
        - Be precise and scientific
        - Use medical terminology with clear explanations
        - Provide balanced, objective information
        - Maintain a supportive and professional tone"""

    def generate_response(self, messages: list) -> str:
        try:
            full_messages = [{"role": "system", "content": self.system_prompt}] + messages
            with st.spinner("Generating response..."):
                completion = self.client.chat.completions.create(
                    model="llama-3.2-11b-vision-preview",  # Updated model name
                    messages=full_messages,
                    temperature=1,
                    max_tokens=1024,
                    top_p=1,
                    stream=False,
                )
            return completion.choices[0].message.content.strip()
        except RateLimitError:
            st.error("Too many requests. Please wait and try again.")
            return "I'm experiencing high traffic. Please try again later."
        except Exception as e:  # Catching all exceptions instead of specific ones
            if isinstance(e, requests.Timeout):
                st.error("The request timed out. Please try again later.")
                return "I'm having trouble connecting. Please check your internet connection."
            elif isinstance(e, APIError):
                st.error(f"An error occurred with the Groq API: {e}")
                return "I apologize, but there was an issue processing your request."
            else:
                st.error(f"An unexpected error occurred: {e}")
                return "An unexpected error occurred. Please try again later."

class PatientRecordManager:
    @staticmethod
    def create_patient_record(name: str, age: int, medical_history: str, conditions: str, medications: str) -> str:
        patient_id = str(uuid.uuid4())[:8]
        record = {
            "id": patient_id,
            "name": name,
            "age": age,
            "medical_history": medical_history,
            "current_conditions": conditions,
            "current_medications": medications,
            "consultations": []
        }
        if "patient_records" not in st.session_state:
            st.session_state.patient_records = {}
        st.session_state.patient_records[patient_id] = record
        return patient_id

def display_message(role: str, content: str):
    role_class = "user-message" if role == "user" else "ai-message"
    avatar = "🧑‍⚕️" if role == "user" else "🤖"
    with st.chat_message(role, avatar=avatar):
        st.markdown(f'<div class="{role_class}">{content}</div>', unsafe_allow_html=True)

def chat_page(chatbot: MedicalAIChatbot):
    st.subheader("Medical Consultation Chat")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.chat_history:
        display_message(message["role"], message["content"])

    user_input = st.chat_input("Ask a medical question or describe symptoms...")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        display_message("user", user_input)
        ai_response = chatbot.generate_response(st.session_state.chat_history)
        st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
        display_message("assistant", ai_response)
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

def patient_records_page():
    st.subheader("Manage Patient Records")
    if "patient_records" not in st.session_state:
        st.session_state.patient_records = {}

    form = st.form(key="patient_form")
    name = form.text_input("Patient Name")
    age = form.number_input("Age", min_value=1)
    medical_history = form.text_area("Medical History")
    conditions = form.text_area("Current Conditions")
    medications = form.text_area("Current Medications")
    submit = form.form_submit_button("Save Record")

    if submit:
        PatientRecordManager.create_patient_record(name, age, medical_history, conditions, medications)
        st.success("Patient record saved successfully!")

    if st.session_state.patient_records:
        for pid, record in st.session_state.patient_records.items():
            with st.expander(record["name"]):
                st.write(f"Age: {record['age']}")
                st.write(f"Medical History: {record['medical_history']}")
                st.write(f"Conditions: {record['current_conditions']}")
                st.write(f"Medications: {record['current_medications']}")

def medical_dashboard():
    st.subheader("Medical Dashboard")
    data = {
        "Total Patients": len(st.session_state.patient_records) if "patient_records" in st.session_state else 0,
        "Consultations Today": 0,  # Placeholder for future implementation
        "Cases Resolved": 0,  # Placeholder for future implementation
    }
    st.write(pd.DataFrame(list(data.items()), columns=["Metric", "Value"]))

def main():
    st.markdown('<div class="main-header"><h1>🩺 MediAssist AI: Your Medical Companion</h1></div>', unsafe_allow_html=True)

    page_options = {
        "Chat Assistant": "💬",
        "Patient Records": "📋",
        "Medical Dashboard": "📊"
    }

    page_labels = list(page_options.keys())
    page_icons = [f'<span style="font-size:20px; margin-right:5px;">{icon}</span>{label}' for label, icon in page_options.items()]

    selected_page = st.sidebar.selectbox(
        "Navigation", page_icons, format_func=lambda x: x
    )

    page_name = selected_page.split('</span>', 1)[1].strip()

    st.sidebar.markdown('<div class="sidebar-content"><h2>MediAssist AI</h2></div>', unsafe_allow_html=True)

    if page_name == "Chat Assistant":
        st.markdown('<div class="stContainer">', unsafe_allow_html=True)
        chat_page(MedicalAIChatbot())
        st.markdown('</div>', unsafe_allow_html=True)
    elif page_name == "Patient Records":
        st.markdown('<div class="stContainer">', unsafe_allow_html=True)
        patient_records_page()
        st.markdown('</div>', unsafe_allow_html=True)
    elif page_name == "Medical Dashboard":
        st.markdown('<div class="stContainer">', unsafe_allow_html=True)
        medical_dashboard()
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()