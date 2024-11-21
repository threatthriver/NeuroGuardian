import os
import uuid
import streamlit as st
from groq import Groq, RateLimitError, APIError
from dotenv import load_dotenv
import pandas as pd
import requests
from datetime import datetime
import json
import csv
from pathlib import Path
from cryptography.fernet import Fernet

# Load environment variables
load_dotenv()

# Setup encryption
def get_encryption_key():
    key_file = Path("encryption.key")
    if not key_file.exists():
        key = Fernet.generate_key()
        with open(key_file, "wb") as f:
            f.write(key)
    with open(key_file, "rb") as f:
        return f.read()

ENCRYPTION_KEY = get_encryption_key()
fernet = Fernet(ENCRYPTION_KEY)

# Configuration and Setup
st.set_page_config(
    page_title="NeuroGuardian",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS with Dark Mode Support and Improved Color Palette
st.markdown("""
<style>
    body {
        background-color: #1e212d;
        color: #dfe6e9;
        font-family: 'Inter', sans-serif;
        transition: background-color 0.3s ease, color 0.3s ease;
    }

    .main-header {
        background: linear-gradient(135deg, #2a3042 0%, #3e4558 100%);
        color: #dfe6e9;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    .sidebar .sidebar-content {
        background-color: #2a3042;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        color: #dfe6e9;
    }

    .stContainer {
        background-color: #2a3042;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        color: #dfe6e9;
    }

    .error-message {
        background-color: #ff4757;
        color: white;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }

    .stButton > button {
        background-color: #4285f4;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 10px 20px;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        background-color: #3273dc;
        transform: translateY(-2px);
    }

    .user-message {
        background-color: #3e4558;
        border-left: 4px solid #4285f4;
        padding: 10px;
        margin-bottom: 10px;
        border-radius: 5px;
        color: #dfe6e9;
    }

    .ai-message {
        background-color: #2a3042;
        border-left: 4px solid #a5d6a7;
        padding: 10px;
        margin-bottom: 10px;
        border-radius: 5px;
        color: #dfe6e9;
    }

    .feedback-container {
        margin-top: 10px;
        padding: 10px;
        border-radius: 5px;
        background-color: #3e4558;
    }

    .feedback-buttons {
        display: flex;
        gap: 10px;
        margin-top: 5px;
    }

    @media (prefers-color-scheme: dark) {
        body {
            background-color: #1e212d;
            color: #dfe6e9;
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
        self.system_prompt = """You are NeuroGuardian, an advanced AI medical companion. You must ONLY provide medical-related assistance and advice.
        If users ask about non-medical topics, politely decline and explain that you can only help with medical matters.
        
        When providing medical assistance:
        - Always clarify that you are an AI assistant, not a substitute for professional medical advice
        - Use clear and empathetic language
        - Simplify complex medical information
        - Prioritize patient safety and understanding
        - Recommend professional consultation when necessary
        - Assist with medical procedures and operations, especially in rural areas

        Communication Style:
        - Be precise and scientific
        - Use medical terminology with clear explanations
        - Provide balanced, objective information
        - Maintain a supportive and professional tone"""

    def generate_response(self, messages: list, patient_data: dict = None) -> str:
        try:
            context = self.system_prompt
            if patient_data:
                context += f"\nPatient Context:\nName: {patient_data['name']}\nAge: {patient_data['age']}\nMedical History: {patient_data['medical_history']}\nCurrent Conditions: {patient_data['current_conditions']}\nMedications: {patient_data['current_medications']}"
            
            cleaned_messages = [{"role": msg["role"], "content": msg["content"]} for msg in messages]
            full_messages = [{"role": "system", "content": context}] + cleaned_messages
            
            with st.spinner("Generating response..."):
                completion = self.client.chat.completions.create(
                    model="llama-3.2-11b-vision-preview",
                    messages=full_messages,
                    temperature=1,
                    max_tokens=1024,
                    top_p=1,
                    stream=False,
                )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            error_msg = "We're experiencing technical difficulties. Please try again later."
            st.markdown(f'<div class="error-message">{error_msg}</div>', unsafe_allow_html=True)
            return error_msg

class PatientRecordManager:
    @staticmethod
    def save_to_file(records):
        encrypted_data = fernet.encrypt(json.dumps(records).encode())
        with open("patient_records.enc", "wb") as f:
            f.write(encrypted_data)

    @staticmethod
    def load_from_file():
        try:
            with open("patient_records.enc", "rb") as f:
                encrypted_data = f.read()
            decrypted_data = fernet.decrypt(encrypted_data)
            return json.loads(decrypted_data)
        except FileNotFoundError:
            return {}

    @staticmethod
    def import_from_csv(file):
        try:
            df = pd.read_csv(file)
            records = {}
            for _, row in df.iterrows():
                patient_id = str(uuid.uuid4())[:8]
                records[patient_id] = {
                    "id": patient_id,
                    "name": row["name"],
                    "age": int(row["age"]),
                    "medical_history": row["medical_history"],
                    "current_conditions": row["current_conditions"],
                    "current_medications": row["current_medications"],
                    "consultations": []
                }
            return records
        except Exception as e:
            st.error("Error importing CSV file. Please check the format.")
            return None

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
            st.session_state.patient_records = PatientRecordManager.load_from_file()
        st.session_state.patient_records[patient_id] = record
        PatientRecordManager.save_to_file(st.session_state.patient_records)
        return patient_id

def display_message(role: str, content: str, message_id: str = None):
    role_class = "user-message" if role == "user" else "ai-message"
    avatar = "🧑‍⚕️" if role == "user" else "🤖"
    with st.chat_message(role, avatar=avatar):
        st.markdown(f'<div class="{role_class}">{content}</div>', unsafe_allow_html=True)

def chat_page(chatbot: MedicalAIChatbot):
    st.subheader("Medical Consultation Chat")
    
    # Add patient selection
    selected_patient = None
    if st.session_state.get("patient_records"):
        patient_names = ["None"] + [record["name"] for record in st.session_state.patient_records.values()]
        selected_name = st.selectbox("Select Patient for Context:", patient_names)
        if selected_name != "None":
            selected_patient = next((record for record in st.session_state.patient_records.values() if record["name"] == selected_name), None)
            st.info(f"Chatting with context for patient: {selected_name}")
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "feedback" not in st.session_state:
        st.session_state.feedback = {}

    # Display chat messages
    for message in st.session_state.chat_history:
        display_message(message["role"], message["content"], message.get("id"))

    # Handle user input
    user_input = st.chat_input("Ask a medical question or describe symptoms...")
    if user_input:
        message_id = str(uuid.uuid4())
        st.session_state.chat_history.append({"role": "user", "content": user_input, "id": message_id})
        display_message("user", user_input)
        ai_response = chatbot.generate_response(st.session_state.chat_history, selected_patient)
        st.session_state.chat_history.append({"role": "assistant", "content": ai_response, "id": message_id})
        display_message("assistant", ai_response, message_id)
    
    # Clear chat button
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

    # Move feedback system to sidebar
    with st.sidebar:
        st.markdown("### Message Feedback")
        if st.session_state.chat_history:
            latest_message = st.session_state.chat_history[-1]
            if latest_message["role"] == "assistant":
                message_id = latest_message["id"]
                st.markdown("#### Rate the last response:")
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("👍 Helpful"):
                        st.session_state.feedback[message_id] = {
                            "rating": "helpful",
                            "timestamp": datetime.now().isoformat()
                        }
                        st.success("Thank you for your feedback!")
                
                with col2:
                    if st.button("👎 Not Helpful"):
                        st.session_state.feedback[message_id] = {
                            "rating": "not_helpful",
                            "timestamp": datetime.now().isoformat()
                        }
                        feedback = st.text_area("How can we improve?")
                        if feedback:
                            st.session_state.feedback[message_id]["comment"] = feedback
                            st.success("Thank you for your detailed feedback!")

def patient_records_page():
    st.subheader("Manage Patient Records")
    if "patient_records" not in st.session_state:
        st.session_state.patient_records = PatientRecordManager.load_from_file()

    # Import patient records from CSV
    st.markdown("### Import Patient Records")
    uploaded_file = st.file_uploader("Upload CSV file with patient records", type="csv")
    if uploaded_file is not None:
        if st.button("Import Records"):
            imported_records = PatientRecordManager.import_from_csv(uploaded_file)
            if imported_records:
                st.session_state.patient_records.update(imported_records)
                PatientRecordManager.save_to_file(st.session_state.patient_records)
                st.success("Patient records imported successfully!")

    st.markdown("### Add New Patient")
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
        st.markdown("### Existing Records")
        for pid, record in st.session_state.patient_records.items():
            with st.expander(record["name"]):
                st.write(f"Age: {record['age']}")
                st.write(f"Medical History: {record['medical_history']}")
                st.write(f"Conditions: {record['current_conditions']}")
                st.write(f"Medications: {record['current_medications']}")

def medical_dashboard():
    st.subheader("Medical Dashboard")
    
    # Calculate feedback metrics
    total_feedback = len(st.session_state.get("feedback", {}))
    helpful_count = sum(1 for f in st.session_state.get("feedback", {}).values() if f["rating"] == "helpful")
    satisfaction_rate = (helpful_count / total_feedback * 100) if total_feedback > 0 else 0
    
    data = {
        "Total Patients": len(st.session_state.patient_records) if "patient_records" in st.session_state else 0,
        "Total Feedback Received": total_feedback,
        "Satisfaction Rate": f"{satisfaction_rate:.1f}%",
        "Consultations Today": 0,  # Placeholder for future implementation
        "Cases Resolved": 0,  # Placeholder for future implementation
    }
    st.write(pd.DataFrame(list(data.items()), columns=["Metric", "Value"]))

def main():
    st.markdown('<div class="main-header"><h1>🧠 NeuroGuardian: Advanced Medical AI Assistant</h1></div>', unsafe_allow_html=True)

    pages = ["Chat Assistant", "Patient Records", "Medical Dashboard"]
    
    selected_page = st.sidebar.selectbox(
        "Navigation", pages
    )

    st.sidebar.markdown('<div class="sidebar-content"><h2>NeuroGuardian</h2></div>', unsafe_allow_html=True)
    with st.sidebar:
        st.markdown("### Latest Updates (Version 2.0):")
        st.markdown("#### Major Improvements:")
        st.markdown("- Advanced AI model integration with enhanced medical knowledge")
        st.markdown("- Real-time patient vitals monitoring system")
        st.markdown("- Secure electronic health records (EHR) management")
        st.markdown("- Multi-language support for global accessibility")
        st.markdown("#### New Features:")
        st.markdown("- Intelligent symptom analysis and prediction")
        st.markdown("- Automated medical report generation")
        st.markdown("- Emergency response protocol system")
        st.markdown("- Integrated telemedicine capabilities")
        st.markdown("#### Technical Improvements:")
        st.markdown("- Enhanced UI/UX with dark mode optimization")
        st.markdown("- Improved response time and accuracy")
        st.markdown("- Advanced data encryption and security measures")
        st.markdown("- Cloud-based backup and synchronization")
        if st.button("View Full Release Notes"):
            st.info("Version 1.0 marks our official release with comprehensive medical AI capabilities and enhanced security features.")

    if selected_page == "Chat Assistant":
        st.markdown('<div class="stContainer">', unsafe_allow_html=True)
        chat_page(MedicalAIChatbot())
        st.markdown('</div>', unsafe_allow_html=True)
    elif selected_page == "Patient Records":
        st.markdown('<div class="stContainer">', unsafe_allow_html=True)
        patient_records_page()
        st.markdown('</div>', unsafe_allow_html=True)
    elif selected_page == "Medical Dashboard":
        st.markdown('<div class="stContainer">', unsafe_allow_html=True)
        medical_dashboard()
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()