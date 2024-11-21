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
import io
import logging
from typing import Dict, List, Optional, Union
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables with error handling
def load_environment():
    try:
        load_dotenv()
        if not os.getenv("GROQ_API_KEY"):
            raise EnvironmentError("GROQ_API_KEY not found in environment variables")
    except Exception as e:
        logger.error(f"Failed to load environment variables: {str(e)}")
        raise

load_environment()

# Setup encryption with better error handling
def get_encryption_key() -> bytes:
    try:
        key_file = Path("encryption.key")
        if not key_file.exists():
            key = Fernet.generate_key()
            key_file.write_bytes(key)
            logger.info("Generated new encryption key")
        return key_file.read_bytes()
    except Exception as e:
        logger.error(f"Failed to get encryption key: {str(e)}")
        raise

try:
    ENCRYPTION_KEY = get_encryption_key()
    fernet = Fernet(ENCRYPTION_KEY)
except Exception as e:
    logger.critical(f"Failed to initialize encryption: {str(e)}")
    raise

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
        try:
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise EnvironmentError("API key not found")
            self.client = Groq(api_key=api_key)
            self._load_system_prompt()
            logger.info("MedicalAIChatbot initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize MedicalAIChatbot: {str(e)}")
            st.error("Failed to initialize chatbot. Please check logs for details.")
            raise

    def _load_system_prompt(self):
        try:
            with open('system_prompt.txt', 'r') as f:
                self.system_prompt = f.read()
        except FileNotFoundError:
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
            logger.warning("System prompt file not found, using default prompt")

    def generate_response(self, messages: List[Dict[str, str]], patient_data: Optional[Dict[str, str]] = None) -> str:
        try:
            context = self.system_prompt
            if patient_data:
                context += self._format_patient_context(patient_data)
            
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
        except RateLimitError:
            error_msg = "Rate limit exceeded. Please try again in a few moments."
            logger.warning("Rate limit exceeded")
            st.warning(error_msg)
            return error_msg
        except APIError as e:
            error_msg = f"API Error: {str(e)}"
            logger.error(f"API Error: {str(e)}")
            st.error(error_msg)
            return error_msg
        except Exception as e:
            error_msg = "An unexpected error occurred. Please try again later."
            logger.error(f"Unexpected error in generate_response: {str(e)}\n{traceback.format_exc()}")
            st.error(error_msg)
            return error_msg

    def _format_patient_context(self, patient_data: Dict[str, str]) -> str:
        return f"\nPatient Context:\nName: {patient_data.get('name', 'N/A')}\nAge: {patient_data.get('age', 'N/A')}\nMedical History: {patient_data.get('medical_history', 'N/A')}\nCurrent Conditions: {patient_data.get('current_conditions', 'N/A')}\nMedications: {patient_data.get('current_medications', 'N/A')}"

class PatientRecordManager:
    @staticmethod
    def save_to_file(records: Dict) -> None:
        try:
            encrypted_data = fernet.encrypt(json.dumps(records).encode())
            backup_path = Path("patient_records.bak")
            file_path = Path("patient_records.enc")
            
            # Create backup of existing file
            if file_path.exists():
                file_path.rename(backup_path)
            
            # Write new data
            with open(file_path, "wb") as f:
                f.write(encrypted_data)
                
            # Remove backup if write was successful
            if backup_path.exists():
                backup_path.unlink()
                
            logger.info("Successfully saved patient records")
        except Exception as e:
            logger.error(f"Failed to save patient records: {str(e)}")
            if backup_path.exists():
                backup_path.rename(file_path)
            raise

    @staticmethod
    def load_from_file() -> Dict:
        try:
            file_path = Path("patient_records.enc")
            if not file_path.exists():
                logger.info("No existing patient records found")
                return {}
                
            with open(file_path, "rb") as f:
                encrypted_data = f.read()
            decrypted_data = fernet.decrypt(encrypted_data)
            records = json.loads(decrypted_data)
            logger.info(f"Successfully loaded {len(records)} patient records")
            return records
        except Exception as e:
            logger.error(f"Failed to load patient records: {str(e)}")
            return {}

    @staticmethod
    def import_from_csv(file) -> Optional[Dict]:
        try:
            content = file.read().decode('utf-8')
            csv_data = csv.DictReader(io.StringIO(content))
            
            required_fields = ["name", "age", "medical_history", "current_conditions", "current_medications"]
            
            headers = csv_data.fieldnames
            if not headers or not all(field in headers for field in required_fields):
                raise ValueError("Invalid CSV format. Missing required columns.")
            
            records = {}
            for row in csv_data:
                try:
                    if not row["name"].strip():
                        continue
                    
                    age = int(row["age"])
                    if age <= 0:
                        continue
                        
                    patient_id = str(uuid.uuid4())[:8]
                    records[patient_id] = {
                        "id": patient_id,
                        "name": row["name"].strip(),
                        "age": age,
                        "medical_history": row["medical_history"].strip(),
                        "current_conditions": row["current_conditions"].strip(),
                        "current_medications": row["current_medications"].strip(),
                        "consultations": []
                    }
                except (ValueError, KeyError) as e:
                    logger.warning(f"Invalid record in CSV: {str(e)}")
                    continue
                    
            if not records:
                raise ValueError("No valid records found in CSV file")
                
            logger.info(f"Successfully imported {len(records)} records from CSV")
            return records
            
        except Exception as e:
            logger.error(f"Failed to import CSV: {str(e)}")
            return None

    @staticmethod
    def create_patient_record(name: str, age: int, medical_history: str, conditions: str, medications: str) -> str:
        try:
            patient_id = str(uuid.uuid4())[:8]
            record = {
                "id": patient_id,
                "name": name,
                "age": age,
                "medical_history": medical_history,
                "current_conditions": conditions,
                "current_medications": medications,
                "consultations": [],
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat()
            }
            
            if "patient_records" not in st.session_state:
                st.session_state.patient_records = PatientRecordManager.load_from_file()
                
            st.session_state.patient_records[patient_id] = record
            PatientRecordManager.save_to_file(st.session_state.patient_records)
            logger.info(f"Created new patient record: {patient_id}")
            return patient_id
        except Exception as e:
            logger.error(f"Failed to create patient record: {str(e)}")
            raise

def display_message(role: str, content: str, message_id: Optional[str] = None) -> None:
    try:
        role_class = "user-message" if role == "user" else "ai-message"
        avatar = "🧑‍⚕️" if role == "user" else "🤖"
        with st.chat_message(role, avatar=avatar):
            st.markdown(f'<div class="{role_class}">{content}</div>', unsafe_allow_html=True)
    except Exception as e:
        logger.error(f"Failed to display message: {str(e)}")
        st.error("Failed to display message")

def chat_page(chatbot: MedicalAIChatbot) -> None:
    try:
        st.subheader("Medical Consultation Chat")
        
        # Initialize session state
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "feedback" not in st.session_state:
            st.session_state.feedback = {}
        
        # Patient selection
        selected_patient = None
        if st.session_state.get("patient_records"):
            patient_names = ["None"] + [record["name"] for record in st.session_state.patient_records.values()]
            selected_name = st.selectbox("Select Patient for Context:", patient_names)
            if selected_name != "None":
                selected_patient = next((record for record in st.session_state.patient_records.values() 
                                      if record["name"] == selected_name), None)
                st.info(f"Chatting with context for patient: {selected_name}")
        
        # Display chat history
        for message in st.session_state.chat_history:
            display_message(message["role"], message["content"], message.get("id"))

        # Handle user input
        user_input = st.chat_input("Ask a medical question or describe symptoms...")
        if user_input:
            message_id = str(uuid.uuid4())
            st.session_state.chat_history.append({
                "role": "user", 
                "content": user_input,
                "id": message_id,
                "timestamp": datetime.now().isoformat()
            })
            display_message("user", user_input)
            
            ai_response = chatbot.generate_response(st.session_state.chat_history, selected_patient)
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": ai_response,
                "id": message_id,
                "timestamp": datetime.now().isoformat()
            })
            display_message("assistant", ai_response, message_id)
        
        # Clear chat button with confirmation
        if st.button("Clear Chat"):
            if st.session_state.chat_history:
                if st.button("Confirm Clear Chat"):
                    st.session_state.chat_history = []
                    st.rerun()
            else:
                st.session_state.chat_history = []
                st.rerun()

        # Feedback system in sidebar
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
                                
    except Exception as e:
        logger.error(f"Error in chat page: {str(e)}\n{traceback.format_exc()}")
        st.error("An error occurred. Please try refreshing the page.")

def patient_records_page() -> None:
    try:
        st.subheader("Manage Patient Records")
        if "patient_records" not in st.session_state:
            st.session_state.patient_records = PatientRecordManager.load_from_file()

        # Import patient records from CSV
        st.markdown("### Import Patient Records")
        st.markdown("""
        Upload a CSV file with the following columns:
        - name (required)
        - age (required, must be positive number)
        - medical_history
        - current_conditions
        - current_medications
        """)
        
        uploaded_file = st.file_uploader("Upload CSV file with patient records", type="csv")
        if uploaded_file is not None:
            if st.button("Import Records"):
                with st.spinner("Importing records..."):
                    imported_records = PatientRecordManager.import_from_csv(uploaded_file)
                    if imported_records:
                        st.session_state.patient_records.update(imported_records)
                        PatientRecordManager.save_to_file(st.session_state.patient_records)
                        st.success(f"Successfully imported {len(imported_records)} patient records!")
                        st.rerun()

        # Add new patient form
        st.markdown("### Add New Patient")
        with st.form(key="patient_form"):
            name = st.text_input("Patient Name")
            age = st.number_input("Age", min_value=1)
            medical_history = st.text_area("Medical History")
            conditions = st.text_area("Current Conditions")
            medications = st.text_area("Current Medications")
            submit = st.form_submit_button("Save Record")

            if submit:
                if not name.strip():
                    st.error("Patient name is required")
                else:
                    try:
                        PatientRecordManager.create_patient_record(
                            name, age, medical_history, conditions, medications
                        )
                        st.success("Patient record saved successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to save patient record: {str(e)}")

        # Display existing records
        if st.session_state.patient_records:
            st.markdown("### Existing Records")
            for pid, record in st.session_state.patient_records.items():
                with st.expander(f"{record['name']} (ID: {pid})"):
                    st.write(f"Age: {record['age']}")
                    st.write(f"Medical History: {record['medical_history']}")
                    st.write(f"Conditions: {record['current_conditions']}")
                    st.write(f"Medications: {record['current_medications']}")
                    if st.button(f"Delete Record {pid}"):
                        if st.button(f"Confirm Delete {pid}"):
                            del st.session_state.patient_records[pid]
                            PatientRecordManager.save_to_file(st.session_state.patient_records)
                            st.success("Record deleted successfully!")
                            st.rerun()
                            
    except Exception as e:
        logger.error(f"Error in patient records page: {str(e)}\n{traceback.format_exc()}")
        st.error("An error occurred. Please try refreshing the page.")

def medical_dashboard() -> None:
    try:
        st.subheader("Medical Dashboard")
        
        # Calculate metrics
        total_feedback = len(st.session_state.get("feedback", {}))
        helpful_count = sum(1 for f in st.session_state.get("feedback", {}).values() 
                          if f["rating"] == "helpful")
        satisfaction_rate = (helpful_count / total_feedback * 100) if total_feedback > 0 else 0
        
        # Get today's consultations
        today = datetime.now().date()
        consultations_today = sum(1 for msg in st.session_state.get("chat_history", [])
                                if msg["role"] == "user" and 
                                datetime.fromisoformat(msg["timestamp"]).date() == today)
        
        data = {
            "Total Patients": len(st.session_state.patient_records) if "patient_records" in st.session_state else 0,
            "Total Feedback Received": total_feedback,
            "Satisfaction Rate": f"{satisfaction_rate:.1f}%",
            "Consultations Today": consultations_today,
            "Active Cases": len([p for p in st.session_state.get("patient_records", {}).values() 
                               if p.get("current_conditions")])
        }
        
        # Display metrics
        df = pd.DataFrame(list(data.items()), columns=["Metric", "Value"])
        st.write(df)
        
        # Add visualizations
        if st.session_state.get("feedback"):
            feedback_df = pd.DataFrame(st.session_state["feedback"].values())
            st.subheader("Feedback Analysis")
            st.bar_chart(feedback_df["rating"].value_counts())
            
    except Exception as e:
        logger.error(f"Error in medical dashboard: {str(e)}\n{traceback.format_exc()}")
        st.error("An error occurred loading the dashboard. Please try refreshing the page.")

def main() -> None:
    try:
        st.markdown('<div class="main-header"><h1>🧠 NeuroGuardian: Advanced Medical AI Assistant</h1></div>', 
                   unsafe_allow_html=True)

        pages = ["Chat Assistant", "Patient Records", "Medical Dashboard"]
        selected_page = st.sidebar.selectbox("Navigation", pages)

        st.sidebar.markdown('<div class="sidebar-content"><h2>NeuroGuardian</h2></div>', 
                          unsafe_allow_html=True)
        
        # Display version info and updates in sidebar
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
                st.info("Version 2.0 introduces comprehensive medical AI capabilities, enhanced security features, and improved user experience.")

        # Route to selected page
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
            
    except Exception as e:
        logger.critical(f"Critical error in main: {str(e)}\n{traceback.format_exc()}")
        st.error("A critical error occurred. Please contact support.")

if __name__ == "__main__":
    main()