from flask import Flask, request, jsonify, render_template
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

app = Flask(__name__)


model_name = "facebook/blenderbot-400M-distill"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


def chatbot_response(user_input):
    inputs = tokenizer(user_input, return_tensors="pt")
    response_ids = model.generate(**inputs, max_new_tokens=100)
    return tokenizer.decode(response_ids[0], skip_special_tokens=True)

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("message", "")
    reply = chatbot_response(user_input)
    return jsonify({"reply": reply})

if __name__ == "__main__":
    app.run(debug=True)