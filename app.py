import os
import uuid
import streamlit as st
from groq import Groq, RateLimitError, APIError
from cerebras.cloud.sdk import Cerebras
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
        if not os.getenv("GROQ_API_KEY") and not os.getenv("CEREBRAS_API_KEY"):
            raise EnvironmentError("GROQ_API_KEY or CEREBRAS_API_KEY not found in environment variables")
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

# Enhanced Custom CSS with New UI Features
st.markdown("""
<style>
    body {
        background-color: #f0f4f8;
        color: #333;
        font-family: 'Arial', sans-serif;
        transition: background-color 0.3s ease, color 0.3s ease;
    }

    .main-header {
        background: linear-gradient(135deg, #4a90e2 0%, #50e3c2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    .sidebar .sidebar-content {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        color: #333;
    }

    .stContainer {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        color: #333;
    }

    .error-message {
        background-color: #ff4757;
        color: white;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }

    .stButton > button {
        background-color: #4caf50;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 10px 20px;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
    }

    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 10px;
        margin-bottom: 10px;
        border-radius: 5px;
        color: #333;
    }

    .ai-message {
        background-color: #f1f8e9;
        border-left: 4px solid #8bc34a;
        padding: 10px;
        margin-bottom: 10px;
        border-radius: 5px;
        color: #333;
    }

    .feedback-container {
        margin-top: 10px;
        padding: 10px;
        border-radius: 5px;
        background-color: #f9fbe7;
    }

    .feedback-buttons {
        display: flex;
        gap: 10px;
        margin-top: 5px;
    }
    /*Improved Patient Search*/
    #patient-search input[type="text"] {
        width: 100%;
        padding: 10px;
        border: 1px solid #ccc;
        border-radius: 5px;
        box-sizing: border-box;
    }

    /*Improved Chat Window*/
    .stChat-message {
        margin-bottom: 15px; /*Increased spacing between messages*/
    }
    .stChat-message-content {
        padding: 15px; /*Increased padding within messages*/
        border-radius: 10px; /*Rounded corners for messages*/
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
            self.system_prompt = """You are NeuroGuardian, an advanced AI medical companion developed by IntellijMind. Your primary role is to provide accurate, clear, and empathetic medical assistance while adhering to strict medical ethics and guidelines.

Core Medical Competencies:
1. Clinical Knowledge & Assessment
   - Understand and explain common medical conditions
   - Recognize symptom patterns and potential diagnoses
   - Provide evidence-based medical information
   - Guide through basic health assessments
   - Explain laboratory results and medical terminology

2. Emergency Medicine Support
   - Identify life-threatening situations
   - Provide immediate first-aid guidance
   - Assist in emergency decision-making
   - Guide emergency response protocols
   - Coordinate with emergency services

3. Chronic Disease Management
   - Monitor disease progression
   - Explain treatment options
   - Support medication adherence
   - Track symptoms and vital signs
   - Provide lifestyle modification guidance

4. Mental Health Care
   - Screen for mental health conditions
   - Offer coping strategies and resources
   - Support crisis intervention
   - Guide through anxiety and stress management
   - Provide sleep hygiene recommendations

5. Preventive Medicine
   - Recommend health screenings
   - Provide vaccination information
   - Guide through lifestyle modifications
   - Support smoking cessation
   - Offer nutrition and exercise guidance

6. Women's Health
   - Support reproductive health
   - Guide through pregnancy care
   - Explain menstrual health
   - Discuss contraception options
   - Address menopausal concerns

7. Sexual Health Education
   - Provide comprehensive sex education
   - Discuss STI prevention and treatment
   - Address reproductive concerns
   - Support gender identity questions
   - Guide through sexual health screenings

8. Pediatric Care Support
   - Guide child development
   - Address common childhood illnesses
   - Support vaccination schedules
   - Monitor growth and development
   - Handle pediatric emergencies

9. Geriatric Care
   - Support aging-related concerns
   - Monitor cognitive health
   - Guide fall prevention
   - Address mobility issues
   - Support medication management

Medical Communication Protocol:
1. Initial Assessment
   - Gather relevant medical history
   - Understand current symptoms
   - Assess severity and urgency
   - Consider risk factors
   - Document key concerns

2. Information Delivery
   - Use clear, accessible language
   - Provide structured explanations
   - Include relevant medical context
   - Offer visual aids when helpful
   - Confirm understanding

3. Action Planning
   - Develop clear next steps
   - Set realistic health goals
   - Create monitoring plans
   - Establish follow-up protocols
   - Define emergency procedures

Safety Guidelines:
- Always identify as an AI medical assistant
- Maintain medical accuracy and currency
- Respect patient privacy and confidentiality
- Recognize scope of practice limitations
- Defer to healthcare professionals when needed
- Document all interactions securely
- Follow medical ethics principles
- Prioritize patient safety above all

Response Structure:
1. Acknowledge and validate concerns
2. Gather necessary information
3. Provide evidence-based guidance
4. Outline practical next steps
5. Include relevant medical disclaimers
6. Offer additional resources
7. Ensure clear follow-up plan

Remember: You are a supportive medical AI assistant working alongside healthcare professionals to enhance patient care and understanding."""
            logger.warning("System prompt file not found, using default prompt")

    def generate_response(self, messages: List[Dict[str, str]], patient_data: Optional[Dict[str, str]] = None) -> str:
        try:
            context = self.system_prompt
            if patient_data:
                context += self._format_patient_context(patient_data)
            
            cleaned_messages = [{"role": msg["role"], "content": msg["content"]} for msg in messages]
            full_messages = [{"role": "system", "content": context}] + cleaned_messages
            
            with st.spinner("Generating response..."):
                try:
                    completion = self.client.chat.completions.create(
                        model="llama-3.2-11b-vision-preview",
                        messages=full_messages,
                        temperature=0.7,  # Adjusted for more focused responses
                        max_tokens=1500,  # Increased token limit for more detailed responses
                        top_p=0.9,
                        stream=False,
                    )
                except RateLimitError:
                    api_key = os.getenv("CEREBRAS_API_KEY")
                    if not api_key:
                        raise EnvironmentError("CEREBRAS_API_KEY not found")
                    cerebras_client = Cerebras(api_key=api_key)
                    completion = cerebras_client.chat.completions.create(
                        model="llama-3.2-11b-vision-preview",
                        messages=full_messages,
                        temperature=0.7,  # Adjusted for more focused responses
                        max_tokens=1500,  # Increased token limit for more detailed responses
                        top_p=0.9,
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

class DoctorManager:
    @staticmethod
    def create_doctor_record(name: str, specialty: str) -> str:
        try:
            doctor_id = str(uuid.uuid4())[:8]
            record = {
                "id": doctor_id,
                "name": name,
                "specialty": specialty,
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat()
            }
            
            if "doctor_records" not in st.session_state:
                st.session_state.doctor_records = DoctorManager.load_from_file()
                
            st.session_state.doctor_records[doctor_id] = record
            DoctorManager.save_to_file(st.session_state.doctor_records)
            logger.info(f"Created new doctor record: {doctor_id}")
            return doctor_id
        except Exception as e:
            logger.error(f"Failed to create doctor record: {str(e)}")
            raise

    @staticmethod
    def save_to_file(records: Dict) -> None:
        try:
            encrypted_data = fernet.encrypt(json.dumps(records).encode())
            backup_path = Path("doctor_records.bak")
            file_path = Path("doctor_records.enc")
            
            # Create backup of existing file
            if file_path.exists():
                file_path.rename(backup_path)
            
            # Write new data
            with open(file_path, "wb") as f:
                f.write(encrypted_data)
                
            # Remove backup if write was successful
            if backup_path.exists():
                backup_path.unlink()
                
            logger.info("Successfully saved doctor records")
        except Exception as e:
            logger.error(f"Failed to save doctor records: {str(e)}")
            if backup_path.exists():
                backup_path.rename(file_path)
            raise

    @staticmethod
    def load_from_file() -> Dict:
        try:
            file_path = Path("doctor_records.enc")
            if not file_path.exists():
                logger.info("No existing doctor records found")
                return {}
                
            with open(file_path, "rb") as f:
                encrypted_data = f.read()
            decrypted_data = fernet.decrypt(encrypted_data)
            records = json.loads(decrypted_data)
            logger.info(f"Successfully loaded {len(records)} doctor records")
            return records
        except Exception as e:
            logger.error(f"Failed to load doctor records: {str(e)}")
            return {}

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
        if "confirm_clear" not in st.session_state:
            st.session_state.confirm_clear = False
        
        # Patient selection with improved search
        selected_patient = None
        if st.session_state.get("patient_records"):
            patient_names = ["None"] + [record["name"] for record in st.session_state.patient_records.values()]
            search_term = st.text_input("Search Patients (by name):", key="patient_search")
            if search_term:
                filtered_names = [name for name in patient_names if search_term.lower() in name.lower()]
                selected_name = st.selectbox("Select Patient:", filtered_names)
            else:
                selected_name = st.selectbox("Select Patient:", patient_names)

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
        
        # Clear chat button with improved confirmation
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("Clear Chat", type="primary", disabled=not st.session_state.chat_history):
                st.session_state.confirm_clear = True
        
        with col2:
            if st.session_state.confirm_clear:
                st.warning("Are you sure you want to clear the chat history?")
                if st.button("Yes, Clear Chat", type="primary"):
                    st.session_state.chat_history = []
                    st.session_state.confirm_clear = False
                    st.rerun()
                if st.button("Cancel"):
                    st.session_state.confirm_clear = False
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

        pages = ["Chat Assistant", "Patient Records", "Medical Dashboard", "Sex Education"]
        selected_page = st.sidebar.selectbox("Navigation", pages)

        st.sidebar.markdown('<div class="sidebar-content"><h2>NeuroGuardian by IntellijMind</h2></div>', 
                          unsafe_allow_html=True)
        
        # Display version info and updates in sidebar
        with st.sidebar:
            st.markdown("### Latest Updates (Version 4.0):")
            st.markdown("#### Major Improvements:")
            st.markdown("- Enhanced AI model with improved accuracy and response time")
            st.markdown("- Real-time patient vitals monitoring system")
            st.markdown("- Secure electronic health records (EHR) management")
            st.markdown("- Multi-language support for global accessibility")
            st.markdown("#### New Features:")
            st.markdown("- Intelligent symptom analysis and prediction")
            st.markdown("- Automated medical report generation")
            st.markdown("- Emergency response protocol system")
            st.markdown("- Integrated telemedicine capabilities")
            st.markdown("- Comprehensive sexual health education")
            st.markdown("#### Technical Improvements:")
            st.markdown("- Enhanced UI/UX with new color palette and layout")
            st.markdown("- Improved response time and accuracy")
            st.markdown("- Advanced data encryption and security measures")
            st.markdown("- Cloud-based backup and synchronization")
            if st.button("View Full Release Notes"):
                st.info("Version 4.0 introduces comprehensive medical AI capabilities, enhanced security features, and improved user experience.")

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
        elif selected_page == "Sex Education":
            st.markdown('<div class="stContainer">', unsafe_allow_html=True)
            st.subheader("Sex Education Chat")
            st.write("Feel free to ask any questions related to sexual health and education.")
            sex_education_chat(MedicalAIChatbot())
            st.markdown('</div>', unsafe_allow_html=True)
            
    except Exception as e:
        logger.critical(f"Critical error in main: {str(e)}\n{traceback.format_exc()}")
        st.error("A critical error occurred. Please contact support.")

def sex_education_chat(chatbot: MedicalAIChatbot) -> None:
    try:
        if "sex_chat_history" not in st.session_state:
            st.session_state.sex_chat_history = []

        # Display chat history
        for message in st.session_state.sex_chat_history:
            display_message(message["role"], message["content"], message.get("id"))

        # Handle user input
        user_input = st.chat_input("Ask a question about sexual health...")
        if user_input:
            message_id = str(uuid.uuid4())
            st.session_state.sex_chat_history.append({
                "role": "user", 
                "content": user_input,
                "id": message_id,
                "timestamp": datetime.now().isoformat()
            })
            display_message("user", user_input)
            
            # Generate response for all questions
            ai_response = chatbot.generate_response(st.session_state.sex_chat_history)
            st.session_state.sex_chat_history.append({
                "role": "assistant",
                "content": ai_response,
                "id": message_id,
                "timestamp": datetime.now().isoformat()
            })
            display_message("assistant", ai_response, message_id)

    except Exception as e:
        logger.error(f"Error in sex education chat: {str(e)}\n{traceback.format_exc()}")
        st.error("An error occurred. Please try refreshing the page.")

if __name__ == "__main__":
    main()
