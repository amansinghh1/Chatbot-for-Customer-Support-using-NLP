from flask import Flask, render_template, request, jsonify
import random, pickle

app = Flask(__name__)

model, vectorizer, intents = pickle.load(open("chatbot/intent_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def chatbot_response():
    user_msg = request.form["msg"]
    X = vectorizer.transform([user_msg])
    intent = model.predict(X)[0]

    for i in intents["intents"]:
        if i["tag"] == intent:
            response = random.choice(i["responses"])
            break

    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
