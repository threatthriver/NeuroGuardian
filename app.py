import os
import streamlit as st
import json
import hashlib
import pickle
from groq import Groq
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

# File path for storing user data
USER_DATA_FILE = 'user_data.pkl'


class NeuroGuardian:
    def __init__(self):
        """Initialize the wellness companion"""
        api_key = os.getenv('GROQ_API_KEY')
        if not api_key:
            st.error("Please set the GROQ_API_KEY in your environment variables")
            st.stop()

        self.client = Groq(api_key=api_key)

        # Configure Streamlit page
        st.set_page_config(page_title="NeuroGuardian | IntelliMind", page_icon="🧠", layout="wide")

        # Initialize session state
        if "logged_in" not in st.session_state:
            st.session_state.logged_in = False
        if "username" not in st.session_state:
            st.session_state.username = None
        if "conversations" not in st.session_state:
            st.session_state.conversations = {}
        if "current_convo" not in st.session_state:
            st.session_state.current_convo = None
        if "whats_new_open" not in st.session_state:
            st.session_state.whats_new_open = False

    def load_user_data(self):
        """Load user data from the persistent file"""
        if os.path.exists(USER_DATA_FILE):
            with open(USER_DATA_FILE, 'rb') as file:
                user_data = pickle.load(file)
                if st.session_state.username in user_data:
                    return user_data[st.session_state.username]
        return None

    def save_user_data(self):
        """Save user data to the persistent file"""
        if not os.path.exists(USER_DATA_FILE):
            with open(USER_DATA_FILE, 'wb') as file:
                pickle.dump({}, file)

        with open(USER_DATA_FILE, 'rb') as file:
            user_data = pickle.load(file)

        user_data[st.session_state.username] = {
            "conversations": st.session_state.conversations
        }

        with open(USER_DATA_FILE, 'wb') as file:
            pickle.dump(user_data, file)

    def delete_account(self):
        """Delete user account and data"""
        if os.path.exists(USER_DATA_FILE):
            with open(USER_DATA_FILE, 'rb') as file:
                user_data = pickle.load(file)

            if st.session_state.username in user_data:
                del user_data[st.session_state.username]

            with open(USER_DATA_FILE, 'wb') as file:
                pickle.dump(user_data, file)

        st.session_state.logged_in = False
        st.session_state.username = None
        st.session_state.conversations = {}
        st.success("Account deleted successfully.")

    def hash_password(self, password):
        """Hash a password with SHA-256"""
        return hashlib.sha256(password.encode('utf-8')).hexdigest()

    def validate_credentials(self, username, password):
        """Validate the username and password (check against stored hash)"""
        if os.path.exists(USER_DATA_FILE):
            with open(USER_DATA_FILE, 'rb') as file:
                user_data = pickle.load(file)

            if username in user_data:
                stored_hash = user_data[username]["password"]
                if stored_hash == self.hash_password(password):
                    return True
        return False

    def register_user(self, username, password):
        """Register a new user"""
        if os.path.exists(USER_DATA_FILE):
            with open(USER_DATA_FILE, 'rb') as file:
                user_data = pickle.load(file)
        else:
            user_data = {}

        if username in user_data:
            st.error("Username already exists. Please choose a different one.")
            return False

        # Save hashed password
        user_data[username] = {"password": self.hash_password(password), "conversations": {}}

        with open(USER_DATA_FILE, 'wb') as file:
            pickle.dump(user_data, file)

        return True

    def generate_ai_response(self, messages):
        """Generate AI response"""
        try:
            messages_for_api = [
                {key: message[key] for key in message if key != "timestamp"}
                for message in messages
            ]

            with st.spinner('Thinking...'):
                completion = self.client.chat.completions.create(
                    model="llama3-groq-70b-8192-tool-use-preview",
                    messages=messages_for_api,
                    temperature=0.6,
                    max_tokens=1500
                )
                return completion.choices[0].message.content
        except Exception as e:
            return f"Sorry, an error occurred: {e}"

    def create_new_conversation(self):
        """Start a new conversation"""
        convo_id = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.conversations[convo_id] = [
            {"role": "system", "content": "You are a compassionate doctor and wellness companion. Provide supportive, constructive guidance for mental health and wellness."}
        ]
        st.session_state.current_convo = convo_id

    def get_conversation_title(self, messages):
        """Generate a title for the conversation"""
        if len(messages) > 1 and messages[1]["role"] == "user":
            return messages[1]["content"][:20] + "..."
        return "Untitled Conversation"

    def show_whats_new(self):
        """Display the What's New popup"""
        st.markdown("""
        ### 🆕 What's New in NeuroGuardian
        **Key Updates:**
        - **Conversation History**: Seamlessly access past chats.
        - **Dynamic Help**: Footer shows support links based on user interaction.
        - **Auto-Generated Titles**: Conversations now have relevant, meaningful titles.
        - **Enhanced What's New Popup**: Stay updated on our latest features.
        """)

    def login(self):
        """Login process"""
        st.session_state.username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if self.validate_credentials(st.session_state.username, password):
                st.session_state.logged_in = True
                user_data = self.load_user_data()
                if user_data:
                    st.session_state.conversations = user_data["conversations"]
                st.success(f"Welcome back, {st.session_state.username}!")
            else:
                st.error("Invalid credentials. Please try again.")

    def register(self):
        """Register process"""
        st.session_state.username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")

        if password != confirm_password:
            st.error("Passwords do not match.")

        if st.button("Register"):
            if self.register_user(st.session_state.username, password):
                st.session_state.logged_in = True
                st.success("Registration successful. You are now logged in!")

    def logout(self):
        """Logout user"""
        st.session_state.logged_in = False
        st.session_state.username = None
        st.session_state.conversations = {}
        st.success("You have logged out successfully.")

    def run(self):
        """Main application runner"""
        st.title("🧠 NeuroGuardian | IntelliMind")
        st.caption("Your AI-Powered Wellness Companion")

        # Check if user is logged in
        if not st.session_state.logged_in:
            auth_method = st.radio("Please choose an option", ["Login", "Register"])
            if auth_method == "Login":
                self.login()
            elif auth_method == "Register":
                self.register()
            return

        # Sidebar for user settings
        with st.sidebar:
            st.header("💡 Conversations")
            st.markdown("Manage your previous chats or start a new one with a single click.")

            # Display past conversations
            if st.session_state.conversations:
                st.markdown("### 🗂️ Chat History")
                for convo_id, messages in st.session_state.conversations.items():
                    title = self.get_conversation_title(messages)
                    if st.button(title, key=convo_id):
                        st.session_state.current_convo = convo_id

            # New conversation button
            if st.button("➕ Start New Conversation"):
                self.create_new_conversation()

            # Divider and "What's New" button
            st.markdown("---")
            if st.button("📣 What's New"):
                st.session_state.whats_new_open = True

            # Delete account button
            if st.button("🗑️ Delete Account"):
                self.delete_account()

            # Logout button
            if st.button("🔓 Logout"):
                self.logout()

        # Show "What's New" popup
        if st.session_state.whats_new_open:
            with st.expander("📣 What's New in NeuroGuardian", expanded=True):
                self.show_whats_new()

        # Ensure an active conversation
        if st.session_state.current_convo is None:
            self.create_new_conversation()

        # Get current conversation
        current_convo_id = st.session_state.current_convo
        messages = st.session_state.conversations[current_convo_id]

        # Chat interface
        st.subheader(f"📝 Current Conversation: {self.get_conversation_title(messages)}")

        # Display chat messages
        for message in messages[1:]:
            with st.chat_message(message["role"]):
                st.markdown(f"**{message['role'].capitalize()}**: {message['content']}")

        # User input for chat
        if prompt := st.chat_input("Type your message here..."):
            messages.append({"role": "user", "content": prompt})

            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate AI response
            with st.chat_message("assistant"):
                ai_response = self.generate_ai_response(messages)
                st.markdown(ai_response)

            # Add AI response to chat
            messages.append({"role": "assistant", "content": ai_response})

        # Save updated conversation
        st.session_state.conversations[current_convo_id] = messages
        self.save_user_data()


def main():
    """Application entry point"""
    app = NeuroGuardian()
    app.run()


if __name__ == "__main__":
    main()
