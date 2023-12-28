from flask import Flask, request, jsonify

from langchain import PromptTemplate, LLMChain, HuggingFaceHub

app = Flask(__name__)

repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
template = """
Task: Generate a short tricky real life word problem from the topic {topic} with options and with complete detailed solution below.
"""

prompt = PromptTemplate(template=template, input_variables=["topic"])

llm = HuggingFaceHub(
    repo_id=repo_id,
    model_kwargs={"temperature": 0.7, "max_length": 1024},
    huggingfacehub_api_token="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"
)
llm_chain = LLMChain(prompt=prompt, llm=llm)

@app.route("/generate_question/", methods=["POST"])
def generate_question():
    if request.headers["Content-Type"] == "application/json":
        data = request.json
    else:
        data = request.form.to_dict()

    topic = data.get("topic")

    if not topic:
        return jsonify({"error": "Topic is required"}), 400

    question = f"Question: {topic}"
    result = llm_chain.run(question)
    return result

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000)
