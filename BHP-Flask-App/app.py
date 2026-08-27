from flask import Flask, render_template, request
import pandas as pd
import pickle
import os


# --------------------------------------------------
# FLASK APP CONFIGURATION
# --------------------------------------------------

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    public_folder=os.path.join(BASE_DIR, "public")
)


# --------------------------------------------------
# FILE PATHS
# --------------------------------------------------

MODEL_PATH = os.path.join(BASE_DIR, "HousePriceModel.pkl")
DATA_PATH = os.path.join(BASE_DIR, "Cleaned_data.csv")


# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

print("Model loaded successfully!")


# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

data = pd.read_csv(DATA_PATH)

# Remove unwanted index column if it exists
if "Unnamed: 0" in data.columns:
    data.drop(columns=["Unnamed: 0"], inplace=True)


# Get unique locations for dropdown
locations = sorted(
    data["location"]
    .dropna()
    .unique()
)

print(f"Total locations loaded: {len(locations)}")


# --------------------------------------------------
# HOME ROUTE
# --------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    error = None

    # Default form values
    # These are used when the page first loads
    form_data = {
        "location": "",
        "bhk": "",
        "bath": "",
        "total_sqft": ""
    }

    # --------------------------------------------------
    # HANDLE FORM SUBMISSION
    # --------------------------------------------------

    if request.method == "POST":

        try:

            # Get submitted form values
            location = request.form.get("location", "")
            bhk = request.form.get("bhk", "")
            bath = request.form.get("bath", "")
            total_sqft = request.form.get("total_sqft", "")

            # Save submitted values
            # This allows them to stay visible after prediction
            form_data = {
                "location": location,
                "bhk": bhk,
                "bath": bath,
                "total_sqft": total_sqft
            }

            # --------------------------------------------------
            # INPUT VALIDATION
            # --------------------------------------------------

            if not location:
                raise ValueError("Please select a location.")

            if not bhk or not bath or not total_sqft:
                raise ValueError("Please fill in all fields.")

            # Convert input values
            bhk_value = int(bhk)
            bath_value = float(bath)
            total_sqft_value = float(total_sqft)

            # Validate positive values
            if bhk_value <= 0:
                raise ValueError("BHK must be greater than 0.")

            if bath_value <= 0:
                raise ValueError(
                    "Number of bathrooms must be greater than 0."
                )

            if total_sqft_value <= 0:
                raise ValueError(
                    "Total square feet must be greater than 0."
                )

            # --------------------------------------------------
            # CREATE MODEL INPUT
            # Column names must match the trained model
            # --------------------------------------------------

            input_data = pd.DataFrame(
                {
                    "location": [location],
                    "total_sqft": [total_sqft_value],
                    "bath": [bath_value],
                    "bhk": [bhk_value]
                }
            )

            print("\nInput Data:")
            print(input_data)

            # --------------------------------------------------
            # MAKE PREDICTION
            # --------------------------------------------------

            predicted_price = model.predict(input_data)[0]

            # Prevent negative prediction
            predicted_price = max(0, predicted_price)

            # Round prediction
            prediction = round(predicted_price, 2)

            print(f"Predicted Price: {prediction} Lakhs")

        except Exception as e:

            print("Prediction Error:", str(e))

            error = str(e)


    # --------------------------------------------------
    # RENDER HTML PAGE
    # --------------------------------------------------

    return render_template(
        "index.html",
        locations=locations,
        prediction=prediction,
        error=error,
        form_data=form_data
    )


# --------------------------------------------------
# RUN FLASK APPLICATION
# --------------------------------------------------

if __name__ == "__main__":

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )
