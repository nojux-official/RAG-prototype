import gradio as gr
import os

# Function to list files in a directory
def list_files(directory):
    try:
        files = os.listdir(directory)
        return files
    except Exception as e:
        return str(e)

with gr.Blocks() as demo:
    file_explorer = gr.FileExplorer(label="Browse Files", root_dir=".", file_count="multiple")
    file_explorer.change(fn=list_files, inputs=file_explorer)

demo.launch()
