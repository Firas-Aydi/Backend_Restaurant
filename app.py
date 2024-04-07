from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import json
import pickle
import joblib

app = Flask(__name__)
CORS(app)
# Charger le modèle de clustering
with open("rest_model2.pkl", "rb") as file:
    rest_model = pickle.load(file)

# Charger le modèle TF-IDF
# tfidf = joblib.load("tfidf_model2.pkl")
with open("tfidf_model2.pkl", "rb") as file:
    tfidf = pickle.load(file)

# Charger les données des restaurants
with open("restaurant_reviews_data.json", "r") as f:
# with open("r.json", "r") as f:
    restaurants = json.load(f)


# Route pour afficher la carte des restaurants avec les avis
# @app.route("/")
# def index():
#     return render_template("index.html", restaurants=json.dumps(restaurants))
@app.route("/")
def get_restaurants():
    return jsonify(restaurants)

@app.route("/predict_sentiment", methods=["POST"])
def predict_sentiment():
    data = request.json
    text = data["text"]
    text_vectorized = tfidf.transform([text])
    sentiment = rest_model.predict(text_vectorized)[0]
    return jsonify({"sentiment": sentiment})


if __name__ == "__main__":
    app.run(debug=True)
