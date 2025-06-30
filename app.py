# app.py
import os
from flask import Flask, request, jsonify
from inference import infer_validity
from sklearn import set_config
set_config(enable_metadata_routing=False)



app = Flask(__name__)

# quick health‐check so you can verify the server is actually up
@app.route("/health", methods=["GET"])
def health():
    return jsonify(status="ok")


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    data = request.get_json(force=True)
    # comment is in data["comment"]
    pred, conf = infer_validity(data["comment"])
    return jsonify(validity=pred, confidence=conf)

    """
# lazy‐load the heavy inference machinery only on demand
_pipeline = None
def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from inference import infer_validity  # loads all your models _here_
        _pipeline = infer_validity
    return _pipeline
"""


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"→ starting Flask on 0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=True)
