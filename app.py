import gradio as gr
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain_community.chat_models import ChatOllama
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()
# Set up the chat model
model = ChatOllama(model="llama3.2", base_url=os.getenv('HOST'))

# Initialize chat history with a system message
system_message = SystemMessage(content="You are a helpful AI assistant.")
chat_history = [system_message]

def chatbot_response(user_input, history=[]):
    # Use ollama to generate a response
    response = model.invoke(user_input)
    bot_reply = response.content

    # Update history
    history.append((user_input, bot_reply))

    return bot_reply, history

# Gradio UI Setup
def gradio_interface():
    with gr.Blocks() as chatbot_ui:
        gr.Markdown("## LangChain with Ollama Chatbot")
        user_input = gr.Textbox(label="User Input", placeholder="Type your message here...", lines=2)
        chat_history = gr.Chatbot()
        send_button = gr.Button("Send")

        def respond(user_input, history):
            bot_reply, history = chatbot_response(user_input, history)
            return history, history

        send_button.click(respond, [user_input, chat_history], [chat_history, chat_history])

    chatbot_ui.launch()

gradio_interface()