import os
from flask import Flask, request, render_template

# Import SimpleDirectoryReader and GPTVectorStoreIndex for indexing.
from llama_index.core import SimpleDirectoryReader, GPTVectorStoreIndex, Settings

# Import our local LLM and embedding model.
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Flask configuration
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Configure local LLM using Ollama (using Llama 3.2 for example)
llm = Ollama(
    model="llama3.2",                   # Specify the local model name
    model_url="http://localhost:11434",  # Ollama server URL (default)
    temperature=0.7,
    max_tokens=512,
    context_window=2048,
)

# Configure a local embedding model using HuggingFace.
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

# Set the global settings for LlamaIndex.
Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 1024  # (Optional) Set desired chunk size for splitting documents.

# Function to process the uploaded document (reads only the specified file)
def process_document(file_path):
    reader = SimpleDirectoryReader(input_files=[file_path])
    docs = reader.load_data()
    # Create the index without an explicit service_context—Settings are used globally.
    index = GPTVectorStoreIndex.from_documents(docs)
    return index.as_query_engine()

# Function to save the uploaded file.
def save_file(file):
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    return filepath

# Main route of the application.
@app.route("/", methods=["GET", "POST"])
def index():
    response_text = ""
    if request.method == "POST":
        file = request.files.get("file")
        question = request.form.get("question", "")
        if file and question:
            file_path = save_file(file)
            query_engine = process_document(file_path)
            response_text = query_engine.query(question).response
    return render_template("index.html", response=response_text)

# Run the Flask application.
if __name__ == "__main__":
    app.run(debug=True)
