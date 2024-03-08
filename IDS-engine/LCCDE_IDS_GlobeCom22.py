from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Define the SMOTE sampling strategy
smote = SMOTE(sampling_strategy={2:1000, 4:1000})

# Apply SMOTE to your data
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train and evaluate a RandomForestClassifier
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, y_train)
rf_predictions = rf_classifier.predict(X_test)
print("Random Forest Classifier:")
print(classification_report(y_test, rf_predictions))

# Train and evaluate an XGBoost classifier
xgb_classifier = XGBClassifier()
xgb_classifier.fit(X_train, y_train)
xgb_predictions = xgb_classifier.predict(X_test)
print("XGBoost Classifier:")
print(classification_report(y_test, xgb_predictions))

# Train and evaluate a LightGBM classifier
lgbm_classifier = LGBMClassifier()
lgbm_classifier.fit(X_train, y_train)
lgbm_predictions = lgbm_classifier.predict(X_test)
print("LightGBM Classifier:")
print(classification_report(y_test, lgbm_predictions))

# Train and evaluate a CatBoost classifier
catboost_classifier = CatBoostClassifier()
catboost_classifier.fit(X_train, y_train)
catboost_predictions = catboost_classifier.predict(X_test)
print("CatBoost Classifier:")
print(classification_report(y_test, catboost_predictions))
