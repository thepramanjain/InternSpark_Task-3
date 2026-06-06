# Titanic Survival Prediction

This workspace contains a notebook-based Titanic survival classifier built around feature engineering, missing-value handling, categorical encoding, model comparison, and explainability.

## Artifacts

- `Titanic_dataset.ipynb` - end-to-end notebook with preprocessing, model training, feature importance, and inference example.
- `titanic_survival_model.joblib` - saved trained pipeline.
- `feature_importance.csv` - ranked feature importances from the random forest model.
- `feature_importance.png` - bar chart of the top imported features.

## What the notebook does

- Extracts passenger `Title` from `Name`.
- Creates `family_size = SibSp + Parch + 1`.
- Converts `Cabin` into a `cabin_present` indicator.
- Imputes missing `Age` values using title and passenger class medians, then falls back to the global age median.
- Imputes missing `Fare` and `Embarked` values.
- Encodes categorical variables with one-hot encoding.
- Trains logistic regression and random forest classifiers.
- Reports validation metrics and feature importance.

## Run the notebook

Open `Titanic_dataset.ipynb` and run all cells.

If you need to recreate the environment manually, install:

```bash
pip install pandas scikit-learn joblib matplotlib seaborn
```

## Inference example

```python
import joblib
import pandas as pd

model = joblib.load("titanic_survival_model.joblib")

sample_passenger = pd.DataFrame([
    {
        "Pclass": 3,
        "Name": "Kelly, Mr. James",
        "Sex": "male",
        "Age": 34.5,
        "SibSp": 0,
        "Parch": 0,
        "Ticket": "330911",
        "Fare": 7.8292,
        "Cabin": None,
        "Embarked": "Q",
    }
])

prediction = int(model.predict(sample_passenger)[0])
probability = float(model.predict_proba(sample_passenger)[0, 1])
print({"predicted_survival": prediction, "survival_probability": round(probability, 3)})
```

The model expects raw passenger attributes. Extra columns such as `PassengerId` are ignored if present.