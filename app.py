import os
import json
from typing import List, Dict
import streamlit as st
from groq import Groq
from dotenv import load_dotenv
from datetime import datetime
import logging
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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
        try:
            return cls(
                role=MessageRole(data["role"]),
                content=data["content"],
                timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else None
            )
        except (KeyError, ValueError) as e:
            logger.error(f"Error parsing message from dict: {e}, Data: {data}")
            return None


class ChatHistory:
    def __init__(self, history_file="chat_history.json"):
        self.messages: List[Message] = []
        self.history_file = history_file
        self.load_from_file()

    def add_message(self, role: MessageRole, content: str):
        message = Message(role=role, content=content, timestamp=datetime.now())
        self.messages.append(message)
        self.save_to_file()

    def clear_history(self):
        # Keep only system messages to reset the chat history
        self.messages = [msg for msg in self.messages if msg.role == MessageRole.SYSTEM]
        self.save_to_file()

    def get_messages_for_api(self) -> List[Dict]:
        return [{"role": msg.role.value, "content": msg.content} for msg in self.messages]

    def save_to_file(self):
        try:
            with open(self.history_file, 'w') as f:
                json.dump([msg.to_dict() for msg in self.messages], f, indent=4)
        except Exception as e:
            logger.error(f"Error saving chat history: {e}")

    def load_from_file(self):
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r') as f:
                    data = json.load(f)
                    self.messages.extend([msg for msg in (Message.from_dict(msg_data) for msg_data in data) if msg])
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Error loading chat history: {e}")


class NeuroGuardian:
    def __init__(self, history_file="chat_history.json"):
        self.validate_environment()
        self.client = self.initialize_client()
        self.configure_page()
        self.initialize_session_state()
        self.chat_history = ChatHistory(history_file)

    def validate_environment(self):
        self.api_key = os.getenv('GROQ_API_KEY')
        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")

    def initialize_client(self) -> Groq:
        try:
            return Groq(api_key=self.api_key)
        except Exception as e:
            raise RuntimeError(f"Error initializing Groq client: {e}") from e

    def configure_page(self):
        st.set_page_config(
            page_title="NeuroGuardian | IntelliMind",
            page_icon="🧠",
            layout="wide",
            initial_sidebar_state="expanded"
        )

    def initialize_session_state(self):
        if "visited" not in st.session_state:
            st.session_state.visited = False
        if "nickname" not in st.session_state:
            st.session_state.nickname = ""
        if "theme" not in st.session_state:
            st.session_state.theme = "light"
        if "show_timestamps" not in st.session_state:
            st.session_state.show_timestamps = False

    def generate_ai_response(self, messages: List[Dict]) -> str:
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
            logger.error(f"Error generating AI response: {e}")
            return "Sorry, I'm having trouble processing your request. Please try again later."

    def render_sidebar(self):
        with st.sidebar:
            st.header("💡 IntelliMind: Empowering Wellness")
            st.subheader("⚙️ User Settings")
            st.session_state.nickname = st.text_input(
                "Your Nickname",
                st.session_state.nickname,
                help="Set your nickname for a personalized experience"
            )
            st.session_state.theme = st.selectbox(
                "Theme",
                ["light", "dark"],
                help="Choose your preferred theme"
            )
            st.session_state.show_timestamps = st.checkbox(
                "Show Message Timestamps",
                st.session_state.show_timestamps
            )

            st.subheader("💬 Chat Management")
            if st.button("🧹 Clear Chat History"):
                self.chat_history.clear_history()
                st.success("Chat history cleared successfully!")

            st.subheader("🌟 Helpful Resources")
            st.markdown("""
                - 📞 **Crisis Hotline**: 988
                - 🌐 [Mental Health Resources](https://www.mentalhealth.gov)
                - 📱 [Wellness Apps Directory](https://mindapps.org)
                - 📚 [Self-Help Library](https://www.helpguide.org)
            """)

    def render_chat_interface(self):
        st.title("🧠 NeuroGuardian | IntelliMind")
        st.caption("Your AI-Powered Wellness Companion")
        if st.session_state.nickname:
            st.write(f"👋 Welcome back, {st.session_state.nickname}! How are you feeling today?")
        else:
            st.write("👋 Welcome! How can I support your mental wellness today?")

        for message in self.chat_history.messages[1:]:
            with st.chat_message(message.role.value):
                st.markdown(message.content)
                if st.session_state.show_timestamps and message.timestamp:
                    st.caption(f"Sent at: {message.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

    def handle_user_input(self):
        if prompt := st.chat_input("Share your thoughts or concerns..."):
            self.chat_history.add_message(MessageRole.USER, prompt)
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                ai_response = self.generate_ai_response(self.chat_history.get_messages_for_api())
                st.markdown(ai_response)
                self.chat_history.add_message(MessageRole.ASSISTANT, ai_response)

    def run(self):
        try:
            self.render_sidebar()
            self.render_chat_interface()
            self.handle_user_input()
        except Exception as e:
            logger.exception("Application error:")
            st.error("An unexpected error occurred. Please refresh the page and try again.")

def main():
    try:
        app = NeuroGuardian()
        app.run()
    except ValueError as e:
        st.error(e)  # Handle missing API Key more gracefully
    except Exception as e:
        logger.exception("Fatal error:")
        st.error("Failed to start the application. Please check the logs and try again.")

if __name__ == "__main__":
    main()
