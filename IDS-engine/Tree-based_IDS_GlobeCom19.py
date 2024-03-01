import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,precision_recall_fscore_support
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from xgboost import plot_importance
from imblearn.over_sampling import SMOTE

#data I need
config = {
    "dataset": "CICIDS2017_sample.csv",
    "features": {
        "features": [], #list of feature. TODO: add all features here
        #deafult to 1.0, paper uses .9. This value removes features until we only use the top 
        #"feature-trimming" features in our model
        "feature_trimming": 1.0 
    },
    #default to this value, if not using SMOTE have empty dictionary
    "SMOTE":{
        #posible key options [0,6]
        #possible value options- any integer
        4: 1500
    },
    #possible model_type values are ["decision tree", "random forest","extra trees", "XGBoost"]
    #is list of the model types user want to use, if multiple models it returns the stacking results
    "model_types": ["decision tree"],

    #only have to give this a value if "XGBoost" is in "model_types"
    "XGBoost_params":{
        "n_estimators": 10 #10 is deafult, user can pick any integer
    }
}

def run(config):
    df = pd.read_csv('./data/CICIDS2017_sample.csv')

    # Min-max normalization
    numeric_features = df.dtypes[df.dtypes != 'object'].index
    df[numeric_features] = df[numeric_features].apply(
        lambda x: (x - x.min()) / (x.max()-x.min()))
    # Fill empty values by 0
    df = df.fillna(0)

    #train test split
    labelencoder = LabelEncoder()
    df.iloc[:, -1] = labelencoder.fit_transform(df.iloc[:, -1])
    X = df.drop(['Label'],axis=1).values 
    y = df.iloc[:, -1].values.reshape(-1,1)
    y=np.ravel(y)
    X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = 0.8, test_size = 0.2, random_state = 0,stratify = y)
    y_train=y_train.astype('int') #have to add this line for stuff to work
    y_test=y_test.astype('int') #have to add this line for stuff to work

    #oversample if in config
    if config['SMOTE'] is not None and config['SMOTE']:
        smote=SMOTE(n_jobs=-1,sampling_strategy={4:1500}) # Create 1500 samples for the minority class "4"
        X_train, y_train = smote.fit_resample(X_train, y_train)
    
    #train the models
    for model_type in config['model_types']:
        if model_type == "decision tree":
            pass
        elif model_type == "random forest":
            pass
        elif model_type == "extra trees":
            pass
        elif model_type == "XGBoost":
            pass

    #do feature selection and retrain models if necessary
    
    #do stacking if necessary
        
    #populate and return results



result = run(config)
print(result)
#what I am returning on every run
result = {
    "accuracy": "",
    "precision": "", 
    "recall": "",
    "F1-score": "",
    "classification_report": "",
    "top_three_features": {
        "overall": [],
        "Dos": [],
        "port_scan": [],
        "brutue_force": [],
        "web_attack": [],
        "botnet": [],
        "infiltration": []
    }
}