import shutil
import gradio as gr
from dotenv import load_dotenv
import os
from openai import OpenAI
import torch
from sentence_transformers import SentenceTransformer, util

from lib.embeddings import convert_pdf_to_text

UPLOAD_DIR = "./uploads"

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

# Update the update_vault function to return updated values
def update_vault(wd="."):
    if os.path.exists(os.path.join(wd, "vault.txt")):
        os.remove(os.path.join(wd, "vault.txt"))

    upload_dir = os.path.join(wd, "uploads")
    if not os.path.exists(upload_dir):
        return [], torch.tensor([])

    for root, _, files in os.walk(upload_dir):
        for file in files:
            if file.lower().endswith(".pdf"):
                convert_pdf_to_text(os.path.join(root, file))

    # TODO: FIX!
    global vault_content, vault_embeddings, vault_embeddings_tensor


    # Load the updated vault content
    vault_content = []
    if os.path.exists("vault.txt"):
        with open("vault.txt", "r", encoding='utf-8') as vault_file:
            vault_content = vault_file.readlines()

    # Encode the updated vault content
    model = SentenceTransformer("all-MiniLM-L6-v2")
    vault_embeddings = model.encode(vault_content) if vault_content else []
    vault_embeddings_tensor = torch.tensor(vault_embeddings)


    return vault_content, vault_embeddings_tensor


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




# Function to handle file uploads and copy them to the target directory
def handle_files(files, directory):
    try:
        os.makedirs(directory, exist_ok=True)

        for file_path in files:
            filename = os.path.basename(file_path)
            destination = os.path.join(directory, filename)
            shutil.copy2(file_path, destination)

        return list_files(directory)
    except Exception as e:
        return str(e)


# Function to list files in a directory
def list_files(directory):
    try:
        files = os.listdir(directory)
        return sorted(files)  # Sort files for better readability
    except Exception as e:
        return str(e)





with gr.Blocks() as demo:
    gr.Markdown("## LangChain with Ollama Chatbot")
    user_input = gr.Textbox(label="User Input", placeholder="Type your message here...", lines=2)
    chat_history = gr.Chatbot()
    send_button = gr.Button("Send")


    def respond(user_input, history):
        system_message = "You are a helpful assistant that is an expert at extracting the most useful information from a given text"
        response = ollama_chat(user_input, system_message, vault_embeddings_tensor, vault_content, model, "llama3")

        bot_reply = response

        # Update history
        history.append((user_input, bot_reply))

        return history, history


    send_button.click(respond, [user_input, chat_history], [chat_history, chat_history])

    # File explorer component
    file_explorer = gr.FileExplorer(
        label="Browse Files",
        root_dir=UPLOAD_DIR,
        file_count="multiple"
    )

    # File upload component
    multiple_files = gr.Files(
        label="Upload Multiple Files",
        file_count="multiple",
        type="filepath",
        file_types=[".pdf"]
    )

    cache_update_button = gr.Button("Update file cache")


    # Handle file uploads and update file explorer
    multiple_files.change(
        fn=handle_files,
        inputs=[multiple_files, gr.State(UPLOAD_DIR)]
    )

    # Update file list when file explorer changes
    file_explorer.change(
        fn=list_files,
        inputs=gr.State(UPLOAD_DIR)
    )

    cache_update_button.click(fn=update_vault)

if __name__ == "__main__":
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    # Configuration for the Ollama API client
    client = OpenAI(
        base_url=os.getenv('HOST') + '/v1',
        api_key='llama3'
    )
    #
    # Load the model and vault content
    model = SentenceTransformer("all-MiniLM-L6-v2")
    vault_content, vault_embeddings_tensor = update_vault()


    demo.launch()
