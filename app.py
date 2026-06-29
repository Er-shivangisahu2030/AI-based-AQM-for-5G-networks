from flask import Flask, render_template, request, jsonify
from predictor import CongestionPredictor

app = Flask(__name__)

predictor = CongestionPredictor()


# -----------------------------
# HTML Interface
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    probability = None

    if request.method == "POST":

        queue_length = float(request.form["queue_length"])
        arrival_rate = float(request.form["arrival_rate"])
        service_rate = float(request.form["service_rate"])
        avg_delay_ms = float(request.form["avg_delay_ms"])
        throughput = float(request.form["throughput"])
        drop_rate = float(request.form["drop_rate"])

        prediction, probability = predictor.predict(
            queue_length,
            arrival_rate,
            service_rate,
            avg_delay_ms,
            throughput,
            drop_rate
        )

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability
    )


# -----------------------------
# REST API for Streamlit
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict_api():

    data = request.get_json()

    prediction, probability = predictor.predict(
        float(data["queue_length"]),
        float(data["arrival_rate"]),
        float(data["service_rate"]),
        float(data["avg_delay_ms"]),
        float(data["throughput"]),
        float(data["drop_rate"])
    )

    return jsonify({
        "prediction": int(prediction),
        "probability": probability
    })


if __name__ == "__main__":
    app.run(debug=False)