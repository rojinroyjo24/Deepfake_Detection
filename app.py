from flask import Flask, render_template, request
import os

from predict_video import predict_video
from database import init_db, insert_prediction, get_all_predictions

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize DB
init_db()

@app.route("/")
def home():
    history = get_all_predictions()
    return render_template("index.html", history=history)

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["video"]

    if file:
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        result, prob = predict_video(filepath)

        # ✅ Save to DB
        insert_prediction(file.filename, result, prob)

        history = get_all_predictions()

        return render_template("index.html",
                               prediction=result,
                               probability=prob,
                               history=history)

    return "Error"

if __name__ == "__main__":
    app.run(debug=True)