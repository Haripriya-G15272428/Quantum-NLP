from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import math

app = Flask(__name__)
CORS(app)

# =====================
# LOAD MODEL
# =====================
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

# =====================
# FUNCTIONS
# =====================
def get_next_token_topk(text, topk=5):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[0, -1, :]
    probs = torch.softmax(logits, dim=-1)
    values, indices = torch.topk(probs, k=topk)
    tokens = tokenizer.convert_ids_to_tokens(indices.tolist())
    return list(zip(tokens, values.tolist()))

def simulated_quantum_runtime(R=50, n=5, k=2.9):
    HR_k = sum([i**(-k) for i in range(1, R+1)])
    HR_k2 = sum([i**(-k/2) for i in range(1, R+1)])
    f = math.log(HR_k2 / (HR_k**0.5)) / math.log(R)
    return {"f": f}

# =====================
# ROUTES
# =====================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data.get("text", "")

    tokens = get_next_token_topk(text)
    quantum = simulated_quantum_runtime()

    return jsonify({
        "tokens": tokens,
        "quantum": quantum
    })

# =====================
# RUN SERVER
# =====================
if __name__ == "__main__":
    app.run(debug=True)