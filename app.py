from flask import Flask, render_template, request
from predictor import CongestionPredictor

app = Flask(__name__)

predictor = CongestionPredictor()


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    probability = None

    if request.method == "POST":

        queue_length = float(
            request.form["queue_length"]
        )

        arrival_rate = float(
            request.form["arrival_rate"]
        )

        service_rate = float(
            request.form["service_rate"]
        )

        avg_delay_ms = float(
            request.form["avg_delay_ms"]
        )

        throughput = float(
            request.form["throughput"]
        )

        drop_rate = float(
            request.form["drop_rate"]
        )

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


if __name__ == "__main__":

    app.run(debug=True)