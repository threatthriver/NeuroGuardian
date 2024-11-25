import os
import uuid
import streamlit as st
from groq import Groq, RateLimitError, APIError
from cerebras.cloud.sdk import Cerebras
from dotenv import load_dotenv
import pandas as pd
import requests
from datetime import datetime, timedelta
import json
import csv
from pathlib import Path
from cryptography.fernet import Fernet
import io
import logging
from typing import Dict, List, Optional, Union
import traceback
from pydantic import BaseModel, EmailStr, constr, validator
from tenacity import retry, stop_after_attempt, wait_exponential
import bleach
import secrets
from functools import wraps
import time
import re

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(process)d - %(thread)d - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler(),
        logging.FileHandler('security.log')  # Separate security log
    ]
)
logger = logging.getLogger(__name__)

# Rate limiting decorator
def rate_limit(max_requests: int, window: int):
    requests_history = {}
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            client_id = st.session_state.get('client_id', 'default')
            
            if client_id not in requests_history:
                requests_history[client_id] = []
            
            # Clean old requests
            requests_history[client_id] = [req_time for req_time in requests_history[client_id] 
                                         if now - req_time < window]
            
            if len(requests_history[client_id]) >= max_requests:
                logger.warning(f"Rate limit exceeded for client {client_id}")
                raise Exception("Rate limit exceeded. Please try again later.")
            
            requests_history[client_id].append(now)
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Secure session management
def init_session():
    if 'session_id' not in st.session_state:
        st.session_state.session_id = secrets.token_urlsafe(32)
    if 'client_id' not in st.session_state:
        st.session_state.client_id = secrets.token_urlsafe(16)
    if 'last_activity' not in st.session_state:
        st.session_state.last_activity = datetime.now()

# Session timeout checker
def check_session_timeout(timeout_minutes: int = 30):
    if 'last_activity' in st.session_state:
        if datetime.now() - st.session_state.last_activity > timedelta(minutes=timeout_minutes):
            # Clear session
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            return True
    st.session_state.last_activity = datetime.now()
    return False

# Input validation models
class PatientRecord(BaseModel):
    name: constr(min_length=2, max_length=100)
    age: int
    medical_history: constr(max_length=5000)
    conditions: constr(max_length=1000)
    medications: constr(max_length=1000)

    @validator('age')
    def validate_age(cls, v):
        if v < 0 or v > 150:
            raise ValueError('Age must be between 0 and 150')
        return v

    @validator('name')
    def sanitize_name(cls, v):
        return bleach.clean(v)

# Enhanced encryption with key rotation
class EncryptionManager:
    def __init__(self, key_file: Path = Path("encryption.key")):
        self.key_file = key_file
        self.key_rotation_interval = timedelta(days=30)
        self.initialize_encryption()

    def initialize_encryption(self):
        try:
            if not self.key_file.exists() or self._should_rotate_key():
                self._generate_new_key()
            self.key = self.key_file.read_bytes()
            self.fernet = Fernet(self.key)
        except Exception as e:
            logger.critical(f"Encryption initialization failed: {str(e)}")
            raise

    def _should_rotate_key(self) -> bool:
        return (self.key_file.stat().st_mtime < 
                (datetime.now() - self.key_rotation_interval).timestamp())

    def _generate_new_key(self):
        key = Fernet.generate_key()
        self.key_file.write_bytes(key)
        logger.info("Generated new encryption key")

    def encrypt(self, data: str) -> bytes:
        return self.fernet.encrypt(data.encode())

    def decrypt(self, data: bytes) -> str:
        return self.fernet.decrypt(data).decode()

# Initialize encryption manager
encryption_manager = EncryptionManager()

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
            self.groq_api_key = os.getenv("GROQ_API_KEY")
            if not self.groq_api_key:
                raise EnvironmentError("GROQ_API_KEY not found")
            
            self.client = Groq(api_key=self.groq_api_key)
            self._load_system_prompt()
            self.request_timeout = 30
            logger.info("MedicalAIChatbot initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize MedicalAIChatbot: {str(e)}")
            raise

    def _load_system_prompt(self):
        try:
            with open('system_prompt.txt', 'r') as f:
                self.system_prompt = f.read().strip()
            logger.info("System prompt loaded successfully")
        except Exception as e:
            logger.error(f"Error loading system prompt: {str(e)}")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def get_response(self, messages: List[Dict[str, str]], context: Optional[Dict] = None) -> str:
        try:
            # Sanitize input
            sanitized_messages = [
                {
                    "role": bleach.clean(msg["role"]),
                    "content": bleach.clean(msg["content"])
                }
                for msg in messages
            ]
            
            # Add system prompt and context
            full_messages = [{"role": "system", "content": self.system_prompt}]
            
            # Add medical context if available
            if context:
                context_prompt = self._format_medical_context(context)
                full_messages.append({"role": "system", "content": context_prompt})
            
            # Add conversation history
            full_messages.extend(sanitized_messages)
            
            # Get response using Llama model
            response = self.client.chat.completions.create(
                model="llama-3.2-11b-vision-preview",
                messages=full_messages,
                temperature=0.7,  # Balanced between creativity and accuracy
                max_tokens=2048,
                top_p=0.9,
                stream=False,
                timeout=self.request_timeout
            )
            
            return response.choices[0].message.content
                
        except Exception as e:
            logger.error(f"Error getting AI response: {str(e)}")
            raise

    def _format_medical_context(self, context: Dict) -> str:
        context_parts = []
        
        if context.get("patient_info"):
            context_parts.append(f"Patient Information:\n{context['patient_info']}")
        
        if context.get("medical_history"):
            context_parts.append(f"Medical History:\n{context['medical_history']}")
        
        if context.get("current_symptoms"):
            context_parts.append(f"Current Symptoms:\n{context['current_symptoms']}")
        
        if context.get("medications"):
            context_parts.append(f"Current Medications:\n{context['medications']}")
        
        if context.get("allergies"):
            context_parts.append(f"Allergies:\n{context['allergies']}")
        
        if context.get("vital_signs"):
            context_parts.append(f"Vital Signs:\n{context['vital_signs']}")
        
        return "\n\n".join(context_parts)

    def analyze_medical_text(self, text: str) -> Dict:
        """Analyzes medical text for key information and entities."""
        try:
            analysis_prompt = f"""Analyze this medical text and extract key information:
            
            Text: {text}
            
            Extract:
            1. Symptoms
            2. Conditions
            3. Medications
            4. Recommendations
            """
            
            response = self.get_response([{"role": "user", "content": analysis_prompt}])
            return self._parse_medical_analysis(response)
        except Exception as e:
            logger.error(f"Error analyzing medical text: {str(e)}")
            return {}

    def _parse_medical_analysis(self, response: str) -> Dict:
        """Parses the AI response into structured medical data."""
        try:
            sections = {
                "symptoms": [],
                "conditions": [],
                "medications": [],
                "recommendations": []
            }
            
            current_section = None
            for line in response.split("\n"):
                line = line.strip()
                if not line:
                    continue
                
                lower_line = line.lower()
                if "symptoms:" in lower_line:
                    current_section = "symptoms"
                elif "conditions:" in lower_line:
                    current_section = "conditions"
                elif "medications:" in lower_line:
                    current_section = "medications"
                elif "recommendations:" in lower_line:
                    current_section = "recommendations"
                elif current_section and line.startswith("-"):
                    sections[current_section].append(line[1:].strip())
            
            return sections
        except Exception as e:
            logger.error(f"Error parsing medical analysis: {str(e)}")
            return {}

class PatientRecordManager:
    @staticmethod
    def save_to_file(records: Dict) -> None:
        try:
            encrypted_data = encryption_manager.encrypt(json.dumps(records))
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
            decrypted_data = encryption_manager.decrypt(encrypted_data)
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
            encrypted_data = encryption_manager.encrypt(json.dumps(records))
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
            decrypted_data = encryption_manager.decrypt(encrypted_data)
            records = json.loads(decrypted_data)
            logger.info(f"Successfully loaded {len(records)} doctor records")
            return records
        except Exception as e:
            logger.error(f"Failed to load doctor records: {str(e)}")
            return {}

def format_markdown_content(content: str) -> str:
    """Format content with enhanced markdown support and medical terminology highlighting."""
    # Format medical terms and measurements
    content = re.sub(r'\b(diagnosis|treatment|symptoms?|medication|prognosis|dosage)\b', r'**\1**', content, flags=re.IGNORECASE)
    content = re.sub(r'(\d+)\s*(mg|ml|g|kg|mm\s*Hg|°[CF]|mmol/L)', r'`\1 \2`', content)
    
    # Format medical references and studies
    content = re.sub(r'\[(Ref|Reference|Study)\s*:\s*([^\]]+)\]', r'> *\1: \2*', content)
    
    # Format important warnings or notes
    content = re.sub(r'(!+\s*Warning:.*?)(?=\n|$)', r'❗ **\1**', content, flags=re.IGNORECASE)
    content = re.sub(r'(Note:.*?)(?=\n|$)', r'📝 *\1*', content, flags=re.IGNORECASE)
    
    # Format lists and bullet points
    content = re.sub(r'(?m)^[-*]\s+(.*?)$', r'• \1', content)
    
    # Enhance medical terms with emojis
    medical_emojis = {
        r'\b(heart|cardiac)\b': '❤️',
        r'\b(brain|neural|neuro)\b': '🧠',
        r'\b(blood)\b': '🩸',
        r'\b(medicine|medication|drug)\b': '💊',
        r'\b(hospital|clinic)\b': '🏥',
        r'\b(doctor|physician)\b': '👨‍⚕️',
        r'\b(exercise|fitness)\b': '🏃‍♂️',
        r'\b(diet|nutrition)\b': '🥗',
        r'\b(sleep)\b': '😴',
        r'\b(stress|anxiety)\b': '😰',
    }
    
    for pattern, emoji in medical_emojis.items():
        content = re.sub(pattern, f'{emoji} \\g<0>', content, flags=re.IGNORECASE)
    
    return content

def display_message(role: str, content: str, message_id: Optional[str] = None) -> None:
    try:
        role_class = "user-message" if role == "user" else "ai-message"
        avatar = "🧑‍⚕️" if role == "user" else "🤖"
        
        # Format the content with enhanced markdown
        formatted_content = format_markdown_content(content)
        
        with st.chat_message(role, avatar=avatar):
            st.markdown(
                f'''
                <div class="{role_class}">
                    <div class="message-content">
                        {formatted_content}
                    </div>
                    {f'<div class="message-id">{message_id}</div>' if message_id else ''}
                </div>
                ''',
                unsafe_allow_html=True
            )
            
    except Exception as e:
        logger.error(f"Failed to display message: {str(e)}")
        st.error("Failed to display message")

def chat_page(chatbot: MedicalAIChatbot):
    try:
        # Add custom CSS for enhanced styling
        st.markdown("""
            <style>
                /* Global Styles */
                @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
                
                * {
                    font-family: 'Inter', sans-serif;
                }
                
                /* Main Container */
                .main {
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }
                
                /* Chat Header */
                .chat-header {
                    background: linear-gradient(135deg, #0A2647, #144272);
                    padding: 25px;
                    border-radius: 20px;
                    color: white;
                    margin-bottom: 30px;
                    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
                    text-align: center;
                }
                
                .chat-header h2 {
                    margin: 0;
                    font-size: 28px;
                    font-weight: 600;
                    letter-spacing: -0.5px;
                }
                
                .chat-header p {
                    margin-top: 10px;
                    opacity: 0.9;
                    font-size: 16px;
                }
                
                /* Chat Messages */
                .stChatMessage {
                    background: white;
                    border-radius: 20px;
                    padding: 20px;
                    margin: 15px 0;
                    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
                    transition: transform 0.2s ease;
                }
                
                .stChatMessage:hover {
                    transform: translateY(-2px);
                    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
                }
                
                .stChatMessage.user {
                    background: linear-gradient(135deg, #E3F2FD, #BBDEFB);
                    margin-left: 50px;
                }
                
                .stChatMessage.assistant {
                    background: white;
                    margin-right: 50px;
                    border-left: 4px solid #2A5298;
                }
                
                /* Message Content */
                .message-content {
                    font-size: 16px;
                    line-height: 1.8;
                    letter-spacing: 0.3px;
                }
                
                /* Text Formatting */
                .stChatMessage strong {
                    color: #1565C0;
                    font-weight: 600;
                    padding: 2px 5px;
                    border-radius: 4px;
                    background: rgba(21, 101, 192, 0.1);
                }
                
                .stChatMessage em {
                    color: #546E7A;
                    font-style: italic;
                }
                
                .stChatMessage code {
                    background: #F5F7F9;
                    padding: 3px 8px;
                    border-radius: 6px;
                    font-family: 'Monaco', monospace;
                    color: #E91E63;
                    font-size: 14px;
                    border: 1px solid rgba(233, 30, 99, 0.2);
                }
                
                /* Blockquotes */
                .stChatMessage blockquote {
                    background: #F8F9FA;
                    border-left: 4px solid #2A5298;
                    margin: 15px 0;
                    padding: 15px 20px;
                    color: #455A64;
                    font-style: italic;
                    border-radius: 0 10px 10px 0;
                }
                
                /* Lists */
                .stChatMessage ul {
                    list-style-type: none;
                    padding-left: 25px;
                }
                
                .stChatMessage li {
                    margin: 12px 0;
                    position: relative;
                    padding-left: 25px;
                }
                
                .stChatMessage li:before {
                    content: "→";
                    color: #2A5298;
                    position: absolute;
                    left: 0;
                    font-weight: bold;
                }
                
                /* Medical Summary */
                .medical-summary {
                    background: linear-gradient(135deg, #F5F7FA, #E4E7EB);
                    border-radius: 15px;
                    padding: 25px;
                    margin: 20px 0;
                    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
                }
                
                .medical-summary h3 {
                    color: #1565C0;
                    font-size: 20px;
                    font-weight: 600;
                    margin-bottom: 15px;
                    padding-bottom: 10px;
                    border-bottom: 2px solid rgba(21, 101, 192, 0.2);
                }
                
                .medical-summary ul {
                    padding-left: 0;
                }
                
                .medical-summary li {
                    background: white;
                    margin: 10px 0;
                    padding: 12px 20px 12px 40px;
                    border-radius: 10px;
                    position: relative;
                    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
                }
                
                .medical-summary li:before {
                    content: "✓";
                    color: #43A047;
                    position: absolute;
                    left: 15px;
                    font-weight: bold;
                }
                
                /* Special Text Styles */
                .warning-text {
                    background: #FFF3E0;
                    color: #E65100;
                    padding: 10px 15px;
                    border-radius: 8px;
                    border-left: 4px solid #E65100;
                    margin: 10px 0;
                }
                
                .note-text {
                    background: #E3F2FD;
                    color: #1565C0;
                    padding: 10px 15px;
                    border-radius: 8px;
                    border-left: 4px solid #1565C0;
                    margin: 10px 0;
                }
                
                /* Typing Indicator */
                .typing-indicator {
                    background: rgba(42, 82, 152, 0.1);
                    padding: 10px 20px;
                    border-radius: 20px;
                    display: inline-block;
                }
                
                .typing-indicator span {
                    display: inline-block;
                    width: 8px;
                    height: 8px;
                    background: #2A5298;
                    border-radius: 50%;
                    margin: 0 3px;
                    animation: typing 1.4s infinite ease-in-out;
                }
                
                .typing-indicator span:nth-child(2) { animation-delay: 0.2s; }
                .typing-indicator span:nth-child(3) { animation-delay: 0.4s; }
                
                @keyframes typing {
                    0%, 100% { transform: translateY(0); }
                    50% { transform: translateY(-6px); }
                }
                
                /* Input Field */
                .stTextInput input {
                    border: 2px solid #E3E8EF;
                    border-radius: 12px;
                    padding: 12px 20px;
                    font-size: 16px;
                    transition: all 0.3s ease;
                }
                
                .stTextInput input:focus {
                    border-color: #2A5298;
                    box-shadow: 0 0 0 3px rgba(42, 82, 152, 0.1);
                }
                
                /* Disclaimer */
                .disclaimer {
                    background: linear-gradient(135deg, #FFF8E1, #FFECB3);
                    border-left: 4px solid #FFA000;
                    padding: 20px;
                    margin-top: 40px;
                    border-radius: 12px;
                    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
                }
                
                .disclaimer p {
                    color: #795548;
                    margin: 0;
                    font-size: 15px;
                    line-height: 1.6;
                }
                
                /* Scrollbar */
                ::-webkit-scrollbar {
                    width: 8px;
                    height: 8px;
                }
                
                ::-webkit-scrollbar-track {
                    background: #F1F1F1;
                    border-radius: 4px;
                }
                
                ::-webkit-scrollbar-thumb {
                    background: #2A5298;
                    border-radius: 4px;
                }
                
                ::-webkit-scrollbar-thumb:hover {
                    background: #1565C0;
                }
            </style>
        """, unsafe_allow_html=True)

        # Welcome message with improved formatting
        welcome_message = """
        👋 Hello! I'm your **NeuroGuardian Medical Assistant**. 
        
        I'm here to help you with:
        • Medical information and guidance
        • Symptom assessment
        • Health recommendations
        • Medical term explanations
        
        📝 *Please note that I provide general medical information only. Always consult healthcare professionals for specific medical advice.*
        
        How can I assist you today?
        """
        
        # Initialize chat history with welcome message
        if "messages" not in st.session_state:
            st.session_state.messages = []
            st.session_state.messages.append({
                "role": "assistant",
                "content": welcome_message,
                "avatar": "🤖"
            })

        # Display chat messages using Streamlit's chat interface
        for message in st.session_state.messages:
            with st.chat_message(message["role"], avatar=message.get("avatar")):
                st.markdown(format_markdown_content(message["content"]))

        # Chat input
        if prompt := st.chat_input("Type your medical question here...", key="chat_input"):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt, "avatar": "🧑‍⚕️"})
            
            # Display user message
            with st.chat_message("user", avatar="🧑‍⚕️"):
                st.markdown(prompt)

            # Display assistant response with typing indicator
            with st.chat_message("assistant", avatar="🤖"):
                message_placeholder = st.empty()
                
                try:
                    with st.spinner(""):
                        # Show typing indicator
                        message_placeholder.markdown("""
                            <div class="typing-indicator">
                                <span></span><span></span><span></span>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Get response from chatbot
                        response = chatbot.get_response(st.session_state.messages)
                        full_response = ""
                        
                        # Simulate streaming response
                        for chunk in response.split():
                            full_response += chunk + " "
                            time.sleep(0.02)
                            message_placeholder.markdown(format_markdown_content(full_response + "▌"))
                        
                        # Display final response
                        message_placeholder.markdown(format_markdown_content(full_response))
                        
                        # Add to chat history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": full_response,
                            "avatar": "🤖"
                        })

                        # Analyze medical content
                        analysis = chatbot.analyze_medical_text(full_response)
                        
                        # Display medical summary if available
                        if any(analysis.values()):
                            with st.expander("📋 Medical Summary", expanded=True):
                                st.markdown('<div class="medical-summary">', unsafe_allow_html=True)
                                
                                if analysis.get("conditions"):
                                    st.markdown("### 🏥 Identified Conditions")
                                    for condition in analysis["conditions"]:
                                        st.markdown(f"* **{condition}**")
                                
                                if analysis.get("medications"):
                                    st.markdown("### 💊 Mentioned Medications")
                                    for med in analysis["medications"]:
                                        st.markdown(f"* `{med}`")
                                
                                if analysis.get("recommendations"):
                                    st.markdown("### 📋 Key Recommendations")
                                    for rec in analysis["recommendations"]:
                                        st.markdown(f"* _{rec}_")
                                
                                st.markdown('</div>', unsafe_allow_html=True)

                except Exception as e:
                    error_msg = "I apologize, but I encountered an error. Please try rephrasing your question."
                    message_placeholder.error(error_msg)
                    logger.error(f"Chat response error: {str(e)}")

        # Medical disclaimer
        st.markdown("""
            <div class="disclaimer">
                <p><strong>⚕️ Medical Disclaimer:</strong> This AI provides general medical information only. 
                Always consult qualified healthcare professionals for medical decisions.</p>
            </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        logger.error(f"Chat page error: {str(e)}")
        st.error("An error occurred. Please refresh the page.")

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
        # Initialize session
        init_session()
        
        # Check session timeout
        if check_session_timeout():
            st.warning("Your session has expired. Please refresh the page.")
            return

        # Set secure headers
        st.markdown("""
            <meta http-equiv="X-Frame-Options" content="DENY">
            <meta http-equiv="X-Content-Type-Options" content="nosniff">
            <meta http-equiv="Strict-Transport-Security" content="max-age=31536000; includeSubDomains">
            <meta http-equiv="Content-Security-Policy" content="default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline';">
        """, unsafe_allow_html=True)

        st.markdown("""
            <div class='main-header'>
                <h1>🧠 NeuroGuardian</h1>
                <p>Your Trusted Medical AI Assistant</p>
            </div>
        """, unsafe_allow_html=True)

        # Initialize chatbot with error handling
        try:
            chatbot = MedicalAIChatbot()
        except Exception as e:
            st.error("Failed to initialize the chatbot. Please try again later.")
            logger.error(f"Chatbot initialization error: {str(e)}")
            return

        # Navigation with rate limiting
        @rate_limit(max_requests=50, window=60)
        def handle_navigation():
            menu = ["Chat", "Patient Records", "Medical Dashboard", "Sex Education"]
            choice = st.sidebar.selectbox("Navigation", menu)

            if choice == "Chat":
                chat_page(chatbot)
            elif choice == "Patient Records":
                patient_records_page()
            elif choice == "Medical Dashboard":
                medical_dashboard()
            elif choice == "Sex Education":
                sex_education_chat(chatbot)

        handle_navigation()

    except Exception as e:
        logger.error(f"Main application error: {str(e)}\n{traceback.format_exc()}")
        st.error("An unexpected error occurred. Please try again later.")

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
            ai_response = chatbot.get_response(st.session_state.sex_chat_history)
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
