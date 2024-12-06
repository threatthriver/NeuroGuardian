# NeuroGuardian AI Chat

A sophisticated AI chatbot built with Streamlit and Cerebras SDK, featuring multiple chat modes, conversation history, and real-time streaming responses.

## Features

- 💬 Interactive chat interface
- 🔄 Multiple chat modes (General, Creative, Technical)
- 📝 Conversation history with timestamps
- ⚡ Real-time response streaming
- 🎨 Clean and modern UI
- 🎛️ Adjustable response parameters
- 🧹 Clear chat functionality

## Setup Instructions

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root and add your Cerebras API key:
   ```
   CEREBRAS_API_KEY=your_api_key_here
   ```

4. Run the application:
   ```bash
   streamlit run chatbot_app.py
   ```

## Chat Modes

- **General**: Balanced responses suitable for everyday conversation
- **Creative**: More imaginative and unique perspectives
- **Technical**: Precise and detailed technical information

## Requirements

- Python 3.8+
- Streamlit
- Cerebras Cloud SDK
- Python-dotenv

## Usage

1. Select a chat mode from the sidebar
2. Adjust the response parameters as needed
3. Type your message in the input field
4. View the AI's response in real-time
5. Clear chat history using the sidebar button when needed
