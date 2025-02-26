# RAG prototype

## Installation

To install the necessary dependencies, run the following commands:

```sh
pip install gradio
pip install ollama
conda install langchain langchain-community
conda install conda-forge::pypdf
```

## Setting Up Ollama

1. Install Ollama.
2. Pull the `llama3` model:
   ```sh
   ollama pull llama3
   ```
3. Serve requests (you may need to allow access from other IPs if required).

## Configuring Ollama Host

Set the Ollama host in the `.env` file:

```
HOST=http://localhost:11434
```

By default, Ollama runs on port `11434`.

## Running the Application

Start the application by running:

```sh
python app.py
```

## Accessing the Application

Once the application is running, open your browser and navigate to the provided URL.

