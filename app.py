from inference import infer_validity
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# your existing imports & infer_validity()...

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
    <p><strong>Validity:</strong> {{ result.validity }}<br>
       <strong>Confidence:</strong> {{ result.confidence|round(2) }}</p>
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
    # if GET just show the form
    if request.method == "GET":
        return render_template_string(FORM, comment=None, result=None)

    # POST: either JSON or form
    if request.is_json:
        data = request.get_json()
        comment = data.get("comment", "")
    else:
        comment = request.form.get("comment", "")

    pred, conf = infer_validity(comment)
    result = {"validity": pred, "confidence": conf}

    # if it was JSON, return JSON
    if request.is_json:
        return jsonify(**result)

    # otherwise re-render the form with the result
    return render_template_string(FORM, comment=comment, result=result)

# existing health endpoint...
@app.route("/health", methods=["GET"])
def health():
    return jsonify(status="ok")
