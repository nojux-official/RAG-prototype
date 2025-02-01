import gradio as gr
import os
import shutil


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


UPLOAD_DIR = "./uploads"

with gr.Blocks() as demo:
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

if __name__ == "__main__":
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    demo.launch()