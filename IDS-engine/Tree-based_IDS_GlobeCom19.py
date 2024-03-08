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
    "model_types": ["XGBoost"],

    #only have to give this a value if "XGBoost" is in "model_types"
    "XGBoost_params":{
        "n_estimators": 10 #10 is deafult, user can pick any integer
    }
}

def run(config):
    df = pd.read_csv('./data/{}'.format(config['dataset']))

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

    def train_models():
        #train the models
        model_results = {}
        for model_type in config['model_types']:
            if model_type == "decision tree":
                if model_type not in model_results:
                    model_results[model_type] = {}
                # Decision tree training and prediction
                dt = DecisionTreeClassifier(random_state = 0)
                dt.fit(X_train,y_train) 
                dt_score=dt.score(X_test,y_test)
                y_predict=dt.predict(X_test)
                y_true=y_test
                
                precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 

                model_results[model_type]['accuracy'] = dt_score
                model_results[model_type]['precision'] = precision
                model_results[model_type]['recall'] = recall
                model_results[model_type]['F1_score'] = fscore
                model_results[model_type]['classification_report'] = classification_report(y_true,y_predict)

                model_results[model_type]['train']=dt.predict(X_train)
                model_results[model_type]['test']=dt.predict(X_test)
                model_results[model_type]['model'] = dt

                feature=(df.drop(['Label'],axis=1)).columns.values
                model_results[model_type]['top_three_features'] = sorted(zip(map(lambda x: round(x, 4), dt.feature_importances_), feature), reverse=True)[:3]

            elif model_type == "random forest":
                if model_type not in model_results:
                    model_results[model_type] = {}
                # Random Forest training and prediction
                rf = RandomForestClassifier(random_state = 0)
                rf.fit(X_train,y_train) 
                rf_score=rf.score(X_test,y_test)
                y_predict=rf.predict(X_test)
                y_true=y_test

                precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
                model_results[model_type]['accuracy'] = rf_score
                model_results[model_type]['precision'] = precision
                model_results[model_type]['recall'] = recall
                model_results[model_type]['F1_score'] = fscore
                model_results[model_type]['classification_report'] = classification_report(y_true,y_predict)

                model_results[model_type]['train']=rf.predict(X_train)
                model_results[model_type]['test']=rf.predict(X_test)
                model_results[model_type]['model'] = rf

                feature=(df.drop(['Label'],axis=1)).columns.values
                model_results[model_type]['top_three_features'] = sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), feature), reverse=True)[:3]

            elif model_type == "extra trees":
                if model_type not in model_results:
                    model_results[model_type] = {}
                # Extra trees training and prediction
                et = ExtraTreesClassifier(random_state = 0)
                et.fit(X_train,y_train) 
                et_score=et.score(X_test,y_test)
                y_predict=et.predict(X_test)
                y_true=y_test

                precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 

                model_results[model_type]['accuracy'] = et_score
                model_results[model_type]['precision'] = precision
                model_results[model_type]['recall'] = recall
                model_results[model_type]['F1_score'] = fscore
                model_results[model_type]['classification_report'] = classification_report(y_true,y_predict)

                model_results[model_type]['train']=et.predict(X_train)
                model_results[model_type]['test']=et.predict(X_test)
                model_results[model_type]['model'] = et

                feature=(df.drop(['Label'],axis=1)).columns.values
                model_results[model_type]['top_three_features'] = sorted(zip(map(lambda x: round(x, 4), et.feature_importances_), feature), reverse=True)[:3]

            elif model_type == "XGBoost":
                if model_type not in model_results:
                    model_results[model_type] = {}
                # XGboost training and prediction
                xg = xgb.XGBClassifier(n_estimators = config['XGBoost_params']['n_estimators'])
                xg.fit(X_train,y_train)
                xg_score=xg.score(X_test,y_test)
                y_predict=xg.predict(X_test)
                y_true=y_test

                precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 

                model_results[model_type]['accuracy'] = xg_score
                model_results[model_type]['precision'] = precision
                model_results[model_type]['recall'] = recall
                model_results[model_type]['F1_score'] = fscore
                model_results[model_type]['classification_report'] = classification_report(y_true,y_predict)

                model_results[model_type]['train']=xg.predict(X_train)
                model_results[model_type]['test']=xg.predict(X_test)
                model_results[model_type]['model'] = xg

                feature=(df.drop(['Label'],axis=1)).columns.values
                model_results[model_type]['top_three_features'] = sorted(zip(map(lambda x: round(x, 4), xg.feature_importances_), feature), reverse=True)[:3]
        return model_results
    
    model_results = train_models()

    #do feature selection and retrain models if necessary
    if config['features']['feature_trimming'] < 1.0:
        avg_feature = [model_results[model]['model'].feature_importances_ for model in model_results.keys()]
        avg_feature = sum(avg_feature) / len(avg_feature)

        feature=(df.drop(['Label'],axis=1)).columns.values
        f_list = sorted(zip(map(lambda x: round(x, 4), avg_feature), feature), reverse=True)
        # Select the important features from top-importance to bottom-importance until the accumulated importance reaches 0.9 (out of 1)
        Sum = 0
        fs = []
        for i in range(0, len(f_list)):
            Sum = Sum + f_list[i][0]
            fs.append(f_list[i][1])
            if Sum>=config['features']['feature_trimming']:
                break  

        X_fs = df[fs].values

        X_train, X_test, y_train, y_test = train_test_split(X_fs,y, train_size = 0.8, test_size = 0.2, random_state = 0,stratify = y)
        y_train=y_train.astype('int') #have to add this line for stuff to work
        y_test=y_test.astype('int') #have to add this line for stuff to work

        #oversample if in config
        if config['SMOTE'] is not None and config['SMOTE']:
            smote=SMOTE(n_jobs=-1,sampling_strategy={4:1500}) # Create 1500 samples for the minority class "4"
            X_train, y_train = smote.fit_resample(X_train, y_train)
        
        #retrain models
        model_results = train_models()
        
    #do stacking if necessary
    if len(config['model_types']) > 1:
        model_results['stacking'] = {}

        base_predictions = {}
        for model_type in config['model_types']:
            base_predictions[model_type] = model_results[model_type]['train'].ravel()
        base_predictions_train = pd.DataFrame(base_predictions)

        for model_type in config['model_types']:
            model_results[model_type]['train'] = model_results[model_type]['train'].reshape(-1,1)
            model_results[model_type]['test'] = model_results[model_type]['test'].reshape(-1,1)

        x_train = np.concatenate(tuple([model_results[model_type]['train'] for model_type in config['model_types']]), axis=1)
        x_test = np.concatenate(tuple([model_results[model_type]['test'] for model_type in config['model_types']]), axis=1)

        stk = xgb.XGBClassifier().fit(x_train, y_train)
        y_predict=stk.predict(x_test)
        y_true=y_test
        stk_score=accuracy_score(y_true,y_predict)
        precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 

        model_results['stacking']['accuracy'] = stk_score
        model_results['stacking']['precision'] = precision
        model_results['stacking']['recall'] = recall
        model_results['stacking']['F1_score'] = fscore
        model_results['stacking']['classification_report'] = classification_report(y_true,y_predict)

        # model_results['stacking']['train']=stk.predict(X_train)
        # model_results['stacking']['test']=stk.predict(X_test)
        # model_results['stacking']['model'] = stk

        feature=(df.drop(['Label'],axis=1)).columns.values
        model_results['stacking']['top_three_features'] = sorted(zip(map(lambda x: round(x, 4), stk.feature_importances_), feature), reverse=True)[:3]
        
    #populate and return results
    result = {
        "accuracy": "",
        "precision": "", 
        "recall": "",
        "F1_score": "",
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
    if len(config['model_types']) > 1:
        result['accuracy'] = model_results['stacking']['accuracy']
        result['precision'] = model_results['stacking']['precision']
        result['recall'] = model_results['stacking']['recall']
        result['F1_score'] = model_results['stacking']['F1_score']
        result['classification_report'] = model_results['stacking']['classification_report']
        result['top_three_features'] = model_results['stacking']['top_three_features']
    else:
        model_type = config['model_types'][0]

        result['accuracy'] = model_results[model_type]['accuracy']
        result['precision'] = model_results[model_type]['precision']
        result['recall'] = model_results[model_type]['recall']
        result['F1_score'] = model_results[model_type]['F1_score']
        result['classification_report'] = model_results[model_type]['classification_report']
        result['top_three_features'] = model_results[model_type]['top_three_features']
    return result

result = run(config)
print(result)
