from dotenv import load_dotenv
load_dotenv()
from flask import Flask, request, render_template
from llama_index.core import SimpleDirectoryReader,  VectorStoreIndex, ServiceContext
from llama_index.llms import openai
import os

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    response = ""
    if request.method == "POST":
        if "file" in request.files:
            file = request.files["file"]
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            reader = SimpleDirectoryReader(input_dir=UPLOAD_FOLDER)
            docs = reader.load_data()
            service_context = ServiceContext.from_defaults(llm=openai)
            index = VectorStoreIndex.from_documents(docs, service_context=service_context)
            query_engine = index.as_query_engine()

            question = request.form.get("question", "")
            if question:
                response = query_engine.query(question).response
    return render_template("index.html", response=response)

if __name__ == "__main__":
    app.run(debug=True)
