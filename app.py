from dotenv import load_dotenv
load_dotenv()

from flask import Flask, request, render_template
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, ServiceContext
from llama_index.llms import openai
import os

# App configuration
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Helper functions
def save_file(file):
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    return filepath

def process_documents():
    reader = SimpleDirectoryReader(input_dir=UPLOAD_FOLDER)
    docs = reader.load_data()
    service_context = ServiceContext.from_defaults(llm=openai)
    index = VectorStoreIndex.from_documents(docs, service_context=service_context)
    return index.as_query_engine()

def answer_question(query_engine, question):
    return query_engine.query(question).response if question else ""


# Main route
@app.route("/", methods=["GET", "POST"])
def index():
    response = ""
    if request.method == "POST":
        file = request.files.get("file")
        if file:
            save_file(file)
            query_engine = process_documents()
            question = request.form.get("question", "")
            response = answer_question(query_engine, question)
    return render_template("index.html", response=response)


# Run the app
if __name__ == "__main__":
    app.run(debug=True)
