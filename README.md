# 🏠 Bangalore House Price Predictor

A machine learning web application that predicts **Bangalore house prices** based on:

- 📍 Location
- 📐 Total Square Feet
- 🛁 Number of Bathrooms
- 🛏️ BHK

The project uses an updated end-to-end machine learning workflow with **EDA, data cleaning, feature engineering, outlier removal, cross-validation, GridSearchCV, model selection, and Flask deployment**.

---

## 📊 Project Visuals

<img width="707" height="461" alt="image" src="https://github.com/user-attachments/assets/662286a5-1704-4900-9e14-f62cd20081ea" />

### Missing Values Analysis

<img width="990" height="490" alt="590ed1f9-43dc-4bba-b4db-11ed88ea7d74" src="https://github.com/user-attachments/assets/3ca8d427-04f3-4de1-bf8a-7b351f27a0c9" />

### House Price Distribution

<img width="990" height="490" alt="60ecf0df-8c5e-4ac2-a5a1-2d1f916181fc" src="https://github.com/user-attachments/assets/c8395941-48b3-4690-a6b8-e872606f05f6" />

### Data Cleaning and Row Reduction

<img width="985" height="490" alt="54a216dd-0b72-4321-9e07-97a837bae5fe" src="https://github.com/user-attachments/assets/34e9201a-f5e2-4453-ac60-925bc19ab84e" />

### Actual vs Predicted Prices

<img width="790" height="590" alt="369d702e-c2ab-47a5-b849-a83599657782" src="https://github.com/user-attachments/assets/f6c1564b-e85b-4211-bf09-457576b737f6" />

---

## 🚀 Project Workflow

```text
Bengaluru_House_Data.csv
        ↓
Initial Data Inspection
        ↓
Exploratory Data Analysis
        ↓
Data Cleaning & Missing Value Handling
        ↓
Feature Engineering
        ↓
Location Processing
        ↓
Outlier Removal
        ↓
Train-Test Split
        ↓
One-Hot Encoding + Standard Scaling
        ↓
5-Fold Cross-Validation
        ↓
GridSearchCV Hyperparameter Tuning
        ↓
Best Model Selection
        ↓
Final Test Evaluation
        ↓
HousePriceModel.pkl
        ↓
Flask Web Application
```

---

## 🧹 Data Preprocessing

The original dataset contains **13,320 rows and 9 columns**.

### Removed Columns

The following columns are removed from the modelling workflow:

- `area_type`
- `availability`
- `society`
- `balcony`

### Missing Value Handling

- `location` → filled using the mode
- `size` → filled using the mode
- `bath` → filled using the median
- Invalid or non-convertible `total_sqft` values → converted to `NaN` and explicitly removed

### Example: Converting `total_sqft`

```python
def convert_range(value):
    value = str(value).strip()
    parts = value.split("-")

    if len(parts) == 2:
        try:
            return (
                float(parts[0]) + float(parts[1])
            ) / 2
        except ValueError:
            return np.nan

    try:
        return float(value)
    except ValueError:
        return np.nan


data["total_sqft"] = data["total_sqft"].apply(convert_range)
data = data.dropna(subset=["total_sqft"]).copy()
```

The updated workflow identified **46 invalid or non-convertible `total_sqft` values** and removed them explicitly.

---

## ⚙️ Feature Engineering

### BHK Extraction

The `size` column is converted into a numerical `bhk` feature.

```python
data["bhk"] = data["size"].str.split().str[0].astype(int)
```

### Price per Square Foot

An intermediate feature is created for outlier analysis:

```python
data["price_per_sqft"] = (
    data["price"] * 100000 / data["total_sqft"]
)
```

This feature is used during data cleaning and is **not included as a final model input**.

### Rare Location Grouping

Locations occurring **10 times or fewer** are grouped into `other`.

---

## 🚫 Outlier Removal

The notebook applies three main rules:

1. Removes properties where:

```text
total_sqft / bhk < 300
```

2. Removes location-wise `price_per_sqft` outliers using the implemented mean and standard deviation rule.

3. Removes selected BHK price inconsistencies where higher-BHK properties have unusually low price-per-square-foot values compared with lower-BHK properties in the same location.

After cleaning, the final modelling dataset contains:

```text
7,402 rows × 5 columns
```

Final columns:

```text
location
total_sqft
bath
bhk
price
```

---

## 🤖 Machine Learning Pipeline

The final predictors are:

```text
location
total_sqft
bath
bhk
```

The target is:

```text
price
```

### Preprocessing

- `OneHotEncoder(handle_unknown="ignore")` for `location`
- `StandardScaler`
- Scikit-learn `Pipeline`

Example pipeline structure:

```python
preprocessor = ColumnTransformer(
    transformers=[
        (
            "location_encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            ["location"]
        )
    ],
    remainder="passthrough"
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("scaler", StandardScaler()),
        ("model", model)
    ]
)
```

---

## 📈 Models Tested

The following regression models are compared:

- Linear Regression
- Lasso Regression
- Ridge Regression

### Validation

The training data uses **5-fold cross-validation** with **R²** as the model-selection metric.

Baseline cross-validation results:

| Model | Mean CV R² |
|---|---:|
| Ridge Regression | 0.834787 |
| Linear Regression | 0.834779 |
| Lasso Regression | 0.825886 |

---

## 🔍 Hyperparameter Tuning

`GridSearchCV` is used to tune the regularisation parameter `alpha` for Lasso and Ridge Regression.

Example:

```python
ridge_grid = GridSearchCV(
    estimator=ridge_pipeline,
    param_grid={
        "model__alpha": [
            0.01, 0.1, 1, 10, 50, 100, 200
        ]
    },
    cv=5,
    scoring="r2",
    n_jobs=-1
)
```

### Best Results

- Best Lasso `alpha` → **0.1**
- Best Ridge `alpha` → **50**
- Best model → **Ridge Regression**

---

## 📊 Final Model Performance

The selected model is evaluated on an independent **20% test set**.

| Metric | Result |
|---|---:|
| R² Score | **0.8919** |
| MAE | **18.2673 Lakhs** |
| RMSE | **32.7799 Lakhs** |

---

## 💾 Saved Model

The final Flask-ready pipeline is saved as:

```text
HousePriceModel.pkl
```

The saved object contains:

```text
Input Data
   ↓
One-Hot Encoding
   ↓
Standard Scaling
   ↓
Ridge Regression
   ↓
Predicted House Price
```

---

## 🌐 Flask Integration

The Flask application should load:

```python
with open("HousePriceModel.pkl", "rb") as file:
    model = pickle.load(file)
```

The prediction input must contain:

```python
input_data = pd.DataFrame({
    "location": ["Whitefield"],
    "total_sqft": [1200],
    "bath": [2],
    "bhk": [2]
})
```

Then:

```python
prediction = model.predict(input_data)
```

The web application can display the predicted house price while keeping the submitted property details visible after prediction.

---

## 🛠️ Technologies Used

- Python
- pandas
- NumPy
- Matplotlib
- scikit-learn
- Flask
- Pickle
- HTML / CSS

---

## 📁 Suggested Project Structure

```text
BHP-Flask-App/
│
├── BHP_Updated.ipynb
├── Bengaluru_House_Data.csv
├── Cleaned_data.csv
├── HousePriceModel.pkl
│
├── app.py
│
├── templates/
│   └── index.html
│
├── static/
│   └── style.css
│
└── README.md
```

---

## ▶️ How to Run

### 1. Install dependencies

```bash
pip install pandas numpy matplotlib scikit-learn flask
```

### 2. Run the updated notebook

Execute:

```text
BHP_Updated.ipynb
```

This generates:

```text
Cleaned_data.csv
HousePriceModel.pkl
```

### 3. Run the Flask application

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

---

<img width="707" height="461" alt="image" src="https://github.com/user-attachments/assets/662286a5-1704-4900-9e14-f62cd20081ea" />



## ✨ Key Improvements in the Updated Version

- Visual EDA added
- Explicit invalid `total_sqft` handling
- Missing-value handling improved
- 5-fold cross-validation added
- `GridSearchCV` added
- Lasso and Ridge hyperparameters tuned
- Automatic best-model selection
- MAE and RMSE added alongside R²
- Final independent test evaluation
- Flask-ready pipeline saved as `HousePriceModel.pkl`

---

## 👨‍💻 Author

**Abhishek Singh**  
B.Tech Computer Science Engineering  
Specialization: Artificial Intelligence & Machine Learning
