from flask import Flask, request, jsonify, render_template, redirect, url_for
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
import math
import os

app = Flask(__name__)
app.secret_key = "secret123"

# --------------------
# LOGIN SYSTEM
# --------------------
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

users = {}

class User(UserMixin):
    def __init__(self, username):
        self.id = username

@login_manager.user_loader
def load_user(user_id):
    if user_id in users:
        return User(user_id)
    return None

# --------------------
# MODEL LAZY LOAD
# --------------------
tokenizer = None
model = None

def load_model():
    global tokenizer, model
    if tokenizer is None or model is None:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        model_name = "sshleifer/tiny-gpt2"   # lightweight for deployment
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.eval()

def get_next_token_topk(text, topk=5):
    import torch

    load_model()

    inputs = tokenizer(text, return_tensors="pt", truncation=True)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[0, -1, :]
    probs = torch.softmax(logits, dim=-1)

    values, indices = torch.topk(probs, k=topk)
    tokens = tokenizer.convert_ids_to_tokens(indices.tolist())

    return list(zip(tokens, values.tolist()))

def simulated_quantum_runtime(R=50, k=2.9):
    HR_k = sum(i**(-k) for i in range(1, R+1))
    HR_k2 = sum(i**(-k/2) for i in range(1, R+1))

    f = math.log(HR_k2 / (HR_k**0.5)) / math.log(R)

    return {"f": round(f, 5)}

# --------------------
# AUTH ROUTES
# --------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        if username in users and users[username] == password:
            login_user(User(username))
            return redirect(url_for("home"))

        return render_template("login.html", error="Invalid credentials")

    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        users[username] = password
        return redirect(url_for("login"))

    return render_template("register.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))

# --------------------
# MAIN ROUTES
# --------------------
@app.route("/")
@login_required
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
@login_required
def predict():
    data = request.get_json()
    text = data.get("text", "")

    tokens = get_next_token_topk(text)
    quantum = simulated_quantum_runtime()

    return jsonify({
        "tokens": tokens,
        "quantum": quantum
    })

# --------------------
# RUN
# --------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)