from flask import Flask, request, render_template
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load trained model & scaler
ridge_model = pickle.load(open("models/ridge.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))

# Feature order MUST match training data
FEATURE_COLS = [
    "LATITUDE",
    "LONGITUDE",
    "LND_SQFOOT",
    "TOT_LVG_AREA",
    "SPEC_FEAT_VAL",
    "RAIL_DIST",
    "OCEAN_DIST",
    "WATER_DIST",
    "CNTR_DIST",
    "SUBCNTR_DI",
    "HWY_DIST",
    "age",
    "avno60plus",
    "month_sold",
    "structure_quality"
]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():

    if request.method == "POST":
        try:
            data = [
                float(request.form["LATITUDE"]),
                float(request.form["LONGITUDE"]),
                float(request.form["LND_SQFOOT"]),
                float(request.form["TOT_LVG_AREA"]),
                float(request.form["SPEC_FEAT_VAL"]),
                float(request.form["RAIL_DIST"]),
                float(request.form["OCEAN_DIST"]),
                float(request.form["WATER_DIST"]),
                float(request.form["CNTR_DIST"]),
                float(request.form["SUBCNTR_DI"]),
                float(request.form["HWY_DIST"]),
                float(request.form["age"]),
                int(request.form["avno60plus"]),
                int(request.form["month_sold"]),
                int(request.form["structure_quality"])
            ]

            input_df = pd.DataFrame([data], columns=FEATURE_COLS)

            scaled_data = scaler.transform(input_df)
            log_price = ridge_model.predict(scaled_data)[0]
            prediction = np.expm1(log_price)

            return render_template(
                "home.html",
                prediction=f"{prediction:,.2f}"
            )

        except Exception as e:
            return render_template("home.html", prediction=f"Error: {e}")

    return render_template("home.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
