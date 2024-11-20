import os
import logging
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from datetime import datetime
from PIL import Image
import io

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
        client = create_groq_client(API_KEY)

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

        try:
            return completion.choices[0].message.content
        except (AttributeError, IndexError) as e:
            logger.error(f"Error accessing response data: {e}")
            return "Sorry, there was an error processing your request."

    except Exception as e:
        logger.error(f"Error during completion request: {e}")
        return "An error occurred while processing your request. Please try again later."

# Function to save feedback
def save_feedback(feedback, conversation):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("feedback.txt", "a") as f:
        f.write(f"[{timestamp}] Feedback: {feedback}\nConversation:\n")
        for msg in conversation:
            f.write(f"{msg['role']}: {msg['content']}\n")
        f.write("\n" + "-"*50 + "\n")
    logger.info("Feedback saved successfully.")

# Function to process uploaded image
def process_image(image_file):
    try:
        # Open the image with PIL
        image = Image.open(image_file)

        # Optional: Convert image to another format (like RGB) for further processing
        image = image.convert("RGB")
        
        # Here, we would typically send the image to the model. For now, we simulate it.
        image_message = {
            "role": "user",
            "content": "What does this medical image show?"
        }
        return image_message
    except Exception as e:
        logger.error(f"Error processing the image: {e}")
        return None

# Streamlit interface
def main():
    st.title("NeuroGuardian - Beta")
    
    # Display instructions
    st.write("Welcome to the medical AI assistant. Ask your questions or upload medical images.")
    
    # Upload medical image (optional)
    image = st.file_uploader("Upload an X-ray or Medical Image", type=["jpg", "png", "jpeg"])
    
    if image:
        # Display the uploaded image
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("Processing image...")

        # Process the image
        image_message = process_image(image)

        if image_message:
            # Simulate sending the image to the model and getting a response
            response = get_completion([image_message])
            st.write(response)
        else:
            st.error("There was an error processing the image. Please try again.")

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
            st.write("Processing your question...")  # Feedback message for the user

            try:
                # Get the response from the AI model
                ai_response = get_completion(st.session_state['messages'])
                
                # Display the AI response
                st.session_state['messages'].append({"role": "assistant", "content": ai_response})

                # Display chat messages
                for msg in st.session_state['messages']:
                    st.chat_message(msg['content'], is_user=msg['role'] == 'user')
            
            except Exception as e:
                st.error("There was an issue getting the response. Please try again.")
        
            # After the conversation ends, show the feedback section
            st.write("### Your Feedback Matters!")
            feedback = st.text_area("Please share your feedback about the assistant:")
            
            if st.button("Submit Feedback"):
                if feedback:
                    try:
                        save_feedback(feedback, st.session_state['messages'])  # Save feedback with conversation context
                        st.success("Thank you for your feedback!")
                        st.session_state['messages'] = []  # Clear chat history after feedback submission
                    except Exception as e:
                        st.error("There was an issue submitting your feedback. Please try again.")

if __name__ == "__main__":
    main()
