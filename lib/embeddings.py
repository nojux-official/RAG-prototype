import os
import PyPDF2
import re


# Function to convert PDF to text and append to vault.txt
def convert_pdf_to_text(file_path):
    with open(file_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        num_pages = len(pdf_reader.pages)
        text = ''
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            if page.extract_text():
                text += page.extract_text() + " "

        # Normalize whitespace and clean up text
        text = re.sub(r'\s+', ' ', text).strip()

        # Split text into chunks by sentences, respecting a maximum chunk size
        sentences = re.split(r'(?<=[.!?]) +', text)  # split on spaces following sentence-ending punctuation
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            # Check if the current sentence plus the current chunk exceeds the limit
            if len(current_chunk) + len(sentence) + 1 < 1000:  # +1 for the space
                current_chunk += (sentence + " ").strip()
            else:
                # When the chunk exceeds 1000 characters, store it and start a new one
                chunks.append(current_chunk)
                current_chunk = sentence + " "
        if current_chunk:  # Don't forget the last chunk!
            chunks.append(current_chunk)
        with open("vault.txt", "a", encoding="utf-8") as vault_file:
            for chunk in chunks:
                # Write each chunk to its own line
                vault_file.write(chunk.strip() + "\n\n")  # Two newlines to separate chunks
        print(f"PDF content appended to vault.txt with each chunk on a separate line.")


import os


def update_vault(wd="."):
    if os.path.exists(os.path.join(wd, "vault.txt")):
        os.remove(os.path.join(wd, "vault.txt"))

    upload_dir = os.path.join(wd, "uploads")
    if not os.path.exists(upload_dir):
        return

    for root, _, files in os.walk(upload_dir):
        for file in files:
            if file.lower().endswith(".pdf"):
                convert_pdf_to_text(os.path.join(root, file))


# Example usage
if __name__ == "__main__":
    wd = os.path.join(os.getcwd(), "../")
    update_vault(wd)