import gradio as gr
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain_community.chat_models import ChatOllama
from dotenv import load_dotenv
import os
from openai import OpenAI
import torch
from sentence_transformers import SentenceTransformer, util

# ANSI escape codes for colors
PINK = '\033[95m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'

# Load environment variables from .env file
load_dotenv()
# Initialize chat history with a system message
# system_message = SystemMessage(content="You are a helpful AI assistant.")
chat_history = []



# Function to open a file and return its contents as a string
def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()


# Function to get relevant context from the vault based on user input
def get_relevant_context(user_input, vault_embeddings, vault_content, model, top_k=3):
    if vault_embeddings.nelement() == 0:  # Check if the tensor has any elements
        return []
    # Encode the user input
    input_embedding = model.encode([user_input])
    # Compute cosine similarity between the input and vault embeddings
    cos_scores = util.cos_sim(input_embedding, vault_embeddings)[0]
    # Adjust top_k if it's greater than the number of available scores
    top_k = min(top_k, len(cos_scores))
    # Sort the scores and get the top-k indices
    top_indices = torch.topk(cos_scores, k=top_k)[1].tolist()
    # Get the corresponding context from the vault
    relevant_context = [vault_content[idx].strip() for idx in top_indices]
    return relevant_context


# Function to interact with the Ollama model
def ollama_chat(user_input, system_message, vault_embeddings, vault_content, model, ollama_model, conversation_history=[]):
    # Get relevant context from the vault
    relevant_context = get_relevant_context(user_input, vault_embeddings, vault_content, model)
    if relevant_context:
        # Convert list to a single string with newlines between items
        context_str = "\n".join(relevant_context)
        print("Context Pulled from Documents: \n\n" + CYAN + context_str + RESET_COLOR)
    else:
        print(CYAN + "No relevant context found." + RESET_COLOR)

    # Prepare the user's input by concatenating it with the relevant context
    user_input_with_context = user_input
    if relevant_context:
        user_input_with_context = context_str + "\n\n" + user_input

    # Append the user's input to the conversation history
    conversation_history.append({"role": "user", "content": user_input_with_context})

    # Create a message history including the system message and the conversation history
    messages = [
        {"role": "system", "content": system_message},
        *conversation_history
    ]

    # Send the completion request to the Ollama model
    response = client.chat.completions.create(
        model=ollama_model,
        messages=messages
    )

    # Append the model's response to the conversation history
    conversation_history.append({"role": "assistant", "content": response.choices[0].message.content})

    # Return the content of the response from the model
    return response.choices[0].message.content




# Configuration for the Ollama API client
client = OpenAI(
    base_url=os.getenv('HOST')+'/v1',
    api_key='llama3'
)
#
# Load the model and vault content
model = SentenceTransformer("all-MiniLM-L6-v2")
vault_content = []
if os.path.exists("vault.txt"):
    with open("vault.txt", "r", encoding='utf-8') as vault_file:
        vault_content = vault_file.readlines()
vault_embeddings = model.encode(vault_content) if vault_content else []

# Convert to tensor and print embeddings
vault_embeddings_tensor = torch.tensor(vault_embeddings)
print("Embeddings for each line in the vault:")
print(vault_embeddings_tensor)

# while True:
#     user_input = input(YELLOW + "Ask a question about your documents (or type 'quit' to exit): " + RESET_COLOR)
#     if user_input.lower() == 'quit':
#         break
#
#     response = ollama_chat(user_input, system_message, vault_embeddings_tensor, vault_content, model, args.model,
#                            conversation_history)
#     print(NEON_GREEN + "Response: \n\n" + response + RESET_COLOR)


def chatbot_response(user_input, history=[]):

    system_message = "You are a helpful assistant that is an expert at extracting the most useful information from a given text"
    response = ollama_chat(user_input, system_message, vault_embeddings_tensor, vault_content, model, "llama3")

    bot_reply = response

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