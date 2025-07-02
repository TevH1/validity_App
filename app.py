from inference import infer_validity
from flask import Flask, request, jsonify, render_template_string
from datetime import datetime
from pymongo import MongoClient
import os
from bson import ObjectId

app = Flask(__name__)
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
client   = MongoClient(MONGO_URI)
db       = client.validity      # my database name
col      = db.predictions  



# a little HTML template
FORM = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Validity Checker</title>
</head>
<body>
  <h1>Comment Validity Checker</h1>

  <form method="post" action="/predict">
    <textarea name="comment" rows="4" cols="50"
      placeholder="Type your comment here…">{{ comment or '' }}</textarea><br>
    <button type="submit">Check Validity</button>
  </form>

  {% if result %}
    <h2>Result</h2>
    <p><strong>Validity:</strong> 
      {% if result.validity == 1 %}
        <strong>AI-written</strong>
      {% else %}
        <strong>Human-written</strong>
      {% endif %}
    </p>
    <p><strong>Confidence:</strong> {{ result.confidence|round(2) }}</p>

    <form method="post" action="/feedback">
      <input type="hidden" name="doc_id" value="{{ result._id }}">
      <p><strong>Was this prediction correct?</strong></p>
      <button name="feedback" value="correct">👍 Yes</button>
      <button name="feedback" value="incorrect">👎 No</button>
    </form>
  {% endif %}

</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    # redirect to /predict or just show link
    return render_template_string(FORM, comment=None, result=None)

@app.route("/predict", methods=["GET", "POST"])
def predict_endpoint():
    if request.method == "GET":
        return render_template_string(FORM, comment=None, result=None)

    # handle POST
    if request.is_json:
        comment = request.get_json().get("comment", "")
    else:
        comment = request.form.get("comment", "")

    pred, conf = infer_validity(comment)
    user = request.args.get("user_id")
    # record into Mongo
    doc = {
      "comment": comment,
      "user_id":     user,
      "validity": bool(pred),
      "confidence": conf,
      "timestamp": datetime.utcnow()
    }
    res = col.insert_one(doc)
    doc["_id"] = res.inserted_id

    result = type("R", (), doc)()  

    if request.is_json:
        return jsonify(validity=pred, confidence=conf)

    return render_template_string(FORM, comment=comment, result=result)






@app.route("/feedback", methods=["POST"])
def feedback_endpoint():
    doc_id   = request.form["doc_id"]
    fb       = request.form["feedback"]  # "correct" or "incorrect"
    col.update_one(
      {"_id": ObjectId(doc_id)},x
      {"$set": {"feedback": fb, "feedback_ts": datetime.utcnow()}}
    )
    return redirect("/predict")

@app.route("/dashboard")
def dashboard():
    user = request.args.get("user_id")
    docs = list(col.find({"user_id": user}))
    return render_template_string(DASHBOARD_TEMPLATE, docs=docs)


# existing health endpoint...
@app.route("/health", methods=["GET"])
def health():
    return jsonify(status="ok")
