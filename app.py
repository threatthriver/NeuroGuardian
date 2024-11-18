import os
import streamlit as st
from groq import Groq
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

class NeuroGuardian:
    def __init__(self):
        """Initialize the wellness companion"""
        # Validate API key
        api_key = os.getenv('GROQ_API_KEY')
        if not api_key:
            st.error("Please set the GROQ_API_KEY in your environment variables")
            st.stop()
        
        # Initialize Groq client
        self.client = Groq(api_key=api_key)

        # Configure Streamlit page
        st.set_page_config(
            page_title="NeuroGuardian | IntelliMind",
            page_icon="🧠",
            layout="centered"
        )

        # Check if it's the user's first visit
        if "visited" not in st.session_state:
            st.session_state.visited = False

        # Initialize or load user nickname
        if "nickname" not in st.session_state:
            st.session_state.nickname = ""

    def generate_ai_response(self, messages):
        """Generate AI response"""
        try:
            # Create a list of messages without timestamps for API call
            messages_for_api = [
                {key: message[key] for key in message if key != "timestamp"}  # Remove timestamp from the message
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

    def run(self):
        """Main application runner"""
        st.title("🧠 NeuroGuardian | IntelliMind")
        st.caption("Your Wellness Companion")

        # Sidebar for user settings and IntelliMind information
        with st.sidebar:
            st.header("💡 IntelliMind: Empowering Wellness")
            st.markdown("""
                **IntelliMind** is a cutting-edge wellness companion designed to provide personalized mental health support and guidance. With the power of advanced AI, we aim to enhance emotional well-being by offering empathetic conversations, insightful guidance, and helpful resources.

                - 🧑‍⚕️ **Mental Health Support**: Offering compassionate, doctor-like assistance.
                - 🌱 **Personalized Care**: Tailored to your unique needs and emotional state.
                - ⚙️ **Innovative AI**: Powered by the latest in artificial intelligence for seamless interactions.
                
                🌐 [Visit IntelliMind Website](https://intellimind.com) for more information!
            """)
            
            # Add more interactive icons or buttons in the sidebar
            st.sidebar.subheader("Settings")
            st.sidebar.write("💬 **Change your nickname or clear chat history.**")

            # First-time visitor pop-up with close option
            if not st.session_state.visited:
                st.session_state.visited = True
                st.sidebar.info(
                    "Welcome to NeuroGuardian! Your AI wellness companion.\n"
                    "Start typing below to begin your journey to better mental health!"
                )

            # Set or change nickname
            st.session_state.nickname = st.text_input(
                "Your Nickname", st.session_state.nickname, help="Set your nickname for a more personalized experience"
            )
            if st.session_state.nickname:
                st.sidebar.write(f"👤 Nickname: {st.session_state.nickname}")

            # Display chat history without timestamps
            st.sidebar.subheader("📝 Chat History")
            if "messages" in st.session_state:
                for message in st.session_state.messages[1:]:
                    st.sidebar.markdown(f"**{message['role'].capitalize()}**: {message['content']}")

            # Clear chat with button interaction
            if st.button("🧹 Clear Chat History"):
                confirm_clear = st.radio("Are you sure you want to clear the chat history?", ("Yes", "No"))
                if confirm_clear == "Yes":
                    st.session_state.messages = [
                        {"role": "system", "content": "You are a compassionate doctor and wellness companion. Provide supportive, constructive guidance for mental health and wellness."}
                    ]
                    st.success("Chat history cleared successfully!")
                elif confirm_clear == "No":
                    st.info("Your chat history is safe.")

        # Initialize session state for messages with a system prompt for doctor role
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "system", "content": "You are a compassionate doctor and wellness companion. Provide supportive, constructive guidance for mental health and wellness."}
            ]

        # Greet the user based on their nickname or a general greeting
        if st.session_state.nickname:
            st.write(f"👋 Hello, {st.session_state.nickname}! How can I assist you today?")
        else:
            st.write("👋 Hello! How can I support you today?")

        # Display chat messages with no timestamps, only the message content
        for message in st.session_state.messages[1:]:
            with st.chat_message(message["role"]):
                st.markdown(f"**{message['role'].capitalize()}**: {message['content']}")

        # User input for chat
        if prompt := st.chat_input("How can I support you today?"):
            # Add user message to chat history without timestamp
            st.session_state.messages.append({
                "role": "user", 
                "content": prompt
            })
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate AI response
            with st.chat_message("assistant"):
                ai_response = self.generate_ai_response(st.session_state.messages)
                st.markdown(ai_response)
            
            # Add AI response to chat history without timestamp
            st.session_state.messages.append({
                "role": "assistant", 
                "content": ai_response
            })

def main():
    """Application entry point"""
    app = NeuroGuardian()
    app.run()

if __name__ == "__main__":
    main()
