import gradio as gr
from ollama import Client
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Initialize the ollama client with environment variables
client = Client(
    host=os.getenv('HOST'),
)

def chatbot_response(user_input, history=[]):
    # Use ollama to generate a response
    response = client.chat(model='llama3.2', messages=[
        {
            'role': 'user',
            'content': user_input,
        },
    ])
    bot_reply = response['message']['content']

    # Update history
    history.append((user_input, bot_reply))

    return bot_reply, history

# Gradio UI Setup
def gradio_interface():
    with gr.Blocks() as chatbot_ui:
        gr.Markdown("## Fake Chatbot UI")
        user_input = gr.Textbox(label="User Input", placeholder="Type your message here...", lines=2)
        chat_history = gr.Chatbot()
        send_button = gr.Button("Send")

        def respond(user_input, history):
            bot_reply, history = chatbot_response(user_input, history)
            return history, history

        send_button.click(respond, [user_input, chat_history], [chat_history, chat_history])

    chatbot_ui.launch()

gradio_interface()