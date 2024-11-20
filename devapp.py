import os
import logging
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from streamlit_chat import message

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Get the API key from the environment
API_KEY = os.getenv("GROQ_API_KEY")

# Ensure the API key is set
if not API_KEY:
    raise ValueError("API_KEY is not set in the .env file")

# Constants
MODEL_NAME = "llama-3.2-11b-vision-preview"
TEMPERATURE = 1
MAX_TOKENS = 1024
TOP_P = 1

# Function to create the Groq client
def create_groq_client(api_key):
    try:
        client = Groq(api_key=api_key)
        return client
    except Exception as e:
        logger.error(f"Failed to create Groq client: {e}")
        raise

# Function to handle chat completions
def get_completion(messages):
    try:
        client = create_groq_client(API_KEY)  # Use the API key for Groq client authentication

        # Create the chat completion
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            top_p=TOP_P,
            stream=False,
            stop=None,
        )

        # Print the response for debugging (optional)
        logger.debug(f"Completion response: {completion}")

        # Ensure you're accessing the correct part of the response using a safe approach
        try:
            # Access the 'choices' attribute and then the 'message' content using dot notation
            return completion.choices[0].message.content
        except (AttributeError, IndexError) as e:
            logger.error(f"Error accessing response data: {e}")
            return "Sorry, there was an error processing your request."

    except Exception as e:
        logger.error(f"Error during completion request: {e}")
        return "An error occurred while processing your request. Please try again later."

# Streamlit interface
def main():
    st.title("NeuroGuardian - Beta")
    
    # Display instructions
    st.write("Welcome to the medical AI assistant. Ask your questions or upload medical images.")
    
    # Upload medical image (optional)
    image = st.file_uploader("Upload an X-ray or Medical Image", type=["jpg", "png", "jpeg"])
    if image:
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("Processing image...")

        # Placeholder for future image processing logic
        # For now, let's simulate sending the image to the model and get a response.
        image_processing_message = {
            "role": "user",
            "content": "What does this X-ray show?"
        }
        response = get_completion([image_processing_message])
        st.write(response)

    # Chat functionality
    if 'messages' not in st.session_state:
        st.session_state['messages'] = [
            {"role": "system", "content": "You are a medical AI assistant."}
        ]
    
    user_input = st.text_input("Ask a medical question:")

    if st.button("Send"):
        if user_input:
            # Add user message to the conversation
            st.session_state['messages'].append({"role": "user", "content": user_input})
            
            # Get the response from the AI model
            ai_response = get_completion(st.session_state['messages'])
            
            # Display the AI response
            st.session_state['messages'].append({"role": "assistant", "content": ai_response})
            
            # Display chat messages
            for msg in st.session_state['messages']:
                message(msg['content'], is_user=msg['role'] == 'user')

if __name__ == "__main__":
    main()
