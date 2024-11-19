import os
import json
from typing import List, Dict, Optional
import streamlit as st
from groq import Groq
from dotenv import load_dotenv
from datetime import datetime
import logging
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class MessageRole(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

@dataclass
class Message:
    role: MessageRole
    content: str
    timestamp: datetime = None

    def to_dict(self) -> Dict:
        return {
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Message':
        return cls(
            role=MessageRole(data["role"]),
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else None
        )

class ChatHistory:
    def __init__(self):
        self.messages: List[Message] = []
        self.load_system_prompt()

    def load_system_prompt(self):
        """Initialize with system prompt"""
        system_prompt = """You are NeuroGuardian, an advanced AI wellness companion with expertise in mental health support. 
        Your approach combines medical knowledge with empathy, providing evidence-based guidance while maintaining a warm, 
        supportive demeanor. While you can offer general wellness advice, you always clarify that you're an AI assistant 
        and encourage seeking professional help for serious concerns. Focus on:
        1. Active listening and validation
        2. Evidence-based coping strategies
        3. Wellness promotion and prevention
        4. Crisis resource awareness
        Remember to maintain appropriate boundaries and prioritize user safety."""
        
        self.add_message(MessageRole.SYSTEM, system_prompt)

    def add_message(self, role: MessageRole, content: str):
        """Add a new message to the history"""
        message = Message(role=role, content=content, timestamp=datetime.now())
        self.messages.append(message)

    def clear_history(self):
        """Clear chat history but retain system prompt"""
        system_message = next((msg for msg in self.messages if msg.role == MessageRole.SYSTEM), None)
        self.messages = [system_message] if system_message else []

    def get_messages_for_api(self) -> List[Dict]:
        """Get messages formatted for API call"""
        return [{"role": msg.role.value, "content": msg.content} for msg in self.messages]

    def save_to_file(self, filename: str):
        """Save chat history to file"""
        try:
            with open(filename, 'w') as f:
                json.dump([msg.to_dict() for msg in self.messages], f)
        except Exception as e:
            logger.error(f"Error saving chat history: {e}")

    def load_from_file(self, filename: str):
        """Load chat history from file"""
        try:
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    data = json.load(f)
                    self.messages = [Message.from_dict(msg) for msg in data]
        except Exception as e:
            logger.error(f"Error loading chat history: {e}")

class NeuroGuardian:
    def __init__(self):
        """Initialize the wellness companion"""
        self.validate_environment()
        self.initialize_client()
        self.configure_page()
        self.initialize_session_state()
        self.chat_history = ChatHistory()

    def validate_environment(self):
        """Validate required environment variables"""
        self.api_key = os.getenv('GROQ_API_KEY')
        if not self.api_key:
            st.error("Please set the GROQ_API_KEY in your environment variables")
            st.stop()

    def initialize_client(self):
        """Initialize the Groq client"""
        try:
            self.client = Groq(api_key=self.api_key)
        except Exception as e:
            logger.error(f"Error initializing Groq client: {e}")
            st.error("Failed to initialize AI client. Please check your API key.")
            st.stop()

    def configure_page(self):
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title="NeuroGuardian | IntelliMind",
            page_icon="🧠",
            layout="wide",
            initial_sidebar_state="expanded"
        )

    def initialize_session_state(self):
        """Initialize session state variables"""
        if "visited" not in st.session_state:
            st.session_state.visited = False
        if "nickname" not in st.session_state:
            st.session_state.nickname = ""
        if "theme" not in st.session_state:
            st.session_state.theme = "light"
        if "show_timestamps" not in st.session_state:
            st.session_state.show_timestamps = False

    def generate_ai_response(self, messages: List[Dict]) -> str:
        """Generate AI response with error handling and retry logic"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with st.spinner('Processing your message...'):
                    completion = self.client.chat.completions.create(
                        model="llama3-groq-70b-8192-tool-use-preview",
                        messages=messages,
                        temperature=0.7,
                        max_tokens=1500,
                        top_p=0.95,
                        presence_penalty=0.6
                    )
                    return completion.choices[0].message.content
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    return "I apologize, but I'm having trouble processing your request. Please try again in a moment."
                st.warning(f"Retrying... Attempt {attempt + 2}/{max_retries}")

    def render_sidebar(self):
        """Render sidebar content"""
        with st.sidebar:
            st.header("💡 IntelliMind: Empowering Wellness")
            
            # User Settings
            st.subheader("⚙️ User Settings")
            st.session_state.nickname = st.text_input(
                "Your Nickname",
                st.session_state.nickname,
                help="Set your nickname for a personalized experience"
            )
            
            # Theme Selection
            st.session_state.theme = st.selectbox(
                "Theme",
                ["light", "dark"],
                help="Choose your preferred theme"
            )
            
            # Display Options
            st.session_state.show_timestamps = st.toggle(
                "Show Message Timestamps",
                st.session_state.show_timestamps
            )
            
            # Chat Management
            st.subheader("💬 Chat Management")
            if st.button("🧹 Clear Chat History"):
                self.handle_clear_chat()
            
            if st.button("💾 Save Chat History"):
                self.chat_history.save_to_file("chat_history.json")
                st.success("Chat history saved successfully!")
            
            if st.button("📂 Load Chat History"):
                self.chat_history.load_from_file("chat_history.json")
                st.success("Chat history loaded successfully!")
            
            # Resources Section
            st.subheader("🌟 Helpful Resources")
            st.markdown("""
                - 📞 **Crisis Hotline**: 988
                - 🌐 [Mental Health Resources](https://www.mentalhealth.gov)
                - 📱 [Wellness Apps Directory](https://mindapps.org)
                - 📚 [Self-Help Library](https://www.helpguide.org)
            """)

    def handle_clear_chat(self):
        """Handle chat clearing with confirmation"""
        clear_chat = st.popup("Clear Chat History?",
                            "Are you sure you want to clear the chat history?",
                            ("Yes", "No"))
        if clear_chat == "Yes":
            self.chat_history.clear_history()
            st.success("Chat history cleared successfully!")
        else:
            st.info("Chat history preserved.")

    def render_chat_interface(self):
        """Render main chat interface"""
        st.title("🧠 NeuroGuardian | IntelliMind")
        st.caption("Your AI-Powered Wellness Companion")

        # Welcome message
        if st.session_state.nickname:
            st.write(f"👋 Welcome back, {st.session_state.nickname}! How are you feeling today?")
        else:
            st.write("👋 Welcome! How can I support your mental wellness today?")

        # Display chat messages
        for message in self.chat_history.messages[1:]:  # Skip system prompt
            with st.chat_message(message.role.value):
                st.markdown(message.content)
                if st.session_state.show_timestamps and message.timestamp:
                    st.caption(f"Sent at: {message.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

    def handle_user_input(self):
        """Handle user input and generate responses"""
        if prompt := st.chat_input("Share your thoughts or concerns..."):
            # Add user message
            self.chat_history.add_message(MessageRole.USER, prompt)
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate and display AI response
            with st.chat_message("assistant"):
                ai_response = self.generate_ai_response(self.chat_history.get_messages_for_api())
                st.markdown(ai_response)
                self.chat_history.add_message(MessageRole.ASSISTANT, ai_response)

    def run(self):
        """Main application runner"""
        try:
            self.render_sidebar()
            self.render_chat_interface()
            self.handle_user_input()
        except Exception as e:
            logger.error(f"Application error: {e}")
            st.error("An unexpected error occurred. Please refresh the page and try again.")

def main():
    """Application entry point"""
    try:
        app = NeuroGuardian()
        app.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        st.error("Failed to start the application. Please check the logs and try again.")

if __name__ == "__main__":
    main()