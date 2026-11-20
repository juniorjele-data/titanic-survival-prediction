# Titanic Survival Prediction

**Junior Jele | Guided machine learning learning project**

Predict whether a Titanic passenger survived using passenger details such as age,
sex, passenger class, fare, and family information. Compare seven classifiers and
study the differences between accuracy, precision, recall, F1, and ROC AUC.

## Results

The dataset contains 891 passengers. An 80/20 stratified split with
`random_state=42` gives 712 training passengers and 179 test passengers.
Precision, recall, and F1 below refer to the survivor class (`Survived = 1`).

| Model | Test accuracy | Survivor F1 | Mean 5-fold CV accuracy |
|---|---:|---:|---:|
| Logistic regression | 84.36% | 0.7971 | 82.44% |
| SVM (RBF) | 81.56% | 0.7442 | 81.88% |
| Random forest | 81.01% | 0.7424 | 81.89% |
| Gradient boosting | 81.01% | 0.7302 | 81.74% |
| Decision tree | 80.45% | 0.7244 | 80.62% |
| Gaussian naive Bayes | 78.21% | 0.7417 | 78.23% |
| KNN (K = 5) | 77.65% | 0.6923 | 80.91% |

Logistic regression was selected for its performance and simplicity. Its test
ROC AUC was approximately 0.8705. It correctly classified 151 of 179 passengers:
96 true negatives, 55 true positives, 14 false positives, and 14 false negatives.
An always-predict-non-survival baseline achieves 61.45% accuracy on this test split.

Comparing `C` values of 0.01, 0.1, 1, 10, and 100 with training-only
cross-validation selected `C = 1`, retaining the original model setting.
The leading CV scores were close; these results do not establish a decisive or
universal advantage for logistic regression.

## Workflow and changes in this guided rebuild

- Split raw passengers before learning preprocessing settings.
- Fill missing ages using training medians grouped by passenger class and sex,
  with a training-wide fallback. Fill missing departure ports with the training mode.
- Engineer family size, alone status, recorded-cabin status, title, age group,
  and training-derived fare quartiles.
- Fit one-hot encoding on training data. Fit scaling on training data for logistic
  regression, KNN, and SVM.
- Relearn preprocessing inside each cross-validation fold.
- Replace obsolete tuning and summary cells with results from the corrected workflow.
- Interpret tree splits and logistic regression weights, then demonstrate inference
  on a fictional passenger using the fitted preprocessing and model.

The checked notebook runs independently of the original `utils.py` and cleaned CSV.
All 31 code cells were executed in order during the final check. Numerical checks
reproduced the selected model's accuracy, CV result, and fictional-passenger prediction.

## Run on Windows with Anaconda

Open Anaconda Prompt, navigate to this extracted folder, and run:

```bash
conda create -n titanic-project python=3.11 -y
conda activate titanic-project
python -m pip install -r requirements.txt
jupyter notebook
```

Open `03_model_building.ipynb` and run its cells from top to bottom.
If you already have a working `titanic` environment, you can use it instead.
Keep `data/titanic.csv` next to the notebook in the layout below.
Results may differ slightly with different library versions or platforms.

## Files

- `03_model_building.ipynb`: checked notebook with saved results and charts.
- `data/titanic.csv`: source passenger dataset used for this comparison.
- `requirements.txt`: Python dependencies.
- `LICENSE`: preserved upstream MIT software license.

## Limitations

- The test set was inspected throughout this educational comparison. It is not an
  untouched final evaluation, and its results should not be presented as one.
- CV and tuning scores were used for selection; no nested CV was performed.
- Engineered features were not compared with an otherwise equivalent model without
  those features, so their incremental benefit was not measured.
- Some inputs are correlated or redundant. Coefficients describe the fitted model,
  not causal effects or a definitive ranking of feature importance.
- Probabilities were not assessed for calibration. The fictional example has no
  known outcome and is an inference demonstration, not an accuracy test.
- This is a historical learning dataset; the model is not intended for real-world
  safety decisions.

## Acknowledgments and authorship

This is a guided rebuild by Junior Jele, adapted from Taimour Karim's
[Beginner-Data-Science-Projects](https://github.com/tkarim45/Beginner-Data-Science-Projects),
with AI-assisted instruction and notebook cleanup. It is not claimed as a project
created entirely from scratch. The upstream MIT copyright and license are preserved
in `LICENSE`.

The included CSV was obtained from the upstream project's
[raw Titanic dataset](https://github.com/tkarim45/Beginner-Data-Science-Projects/blob/main/Classification/Titanic%20Survival%20Prediction/data/titanic.csv).
The software license does not establish separate ownership of the historical data.
