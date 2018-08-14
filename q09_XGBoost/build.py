# %load q09_XGBoost/build.py
import pandas as pd
import numpy as np
import sys,os
sys.path.append(os.path.join(os.path.dirname(os.curdir)))
from greyatomlib.game_of_thrones.q01_feature_engineering.build import q01_feature_engineering
from greyatomlib.game_of_thrones.q08_preprocessing.build import q08_preprocessing
from xgboost import plot_importance
from sklearn.metrics import roc_auc_score,accuracy_score
from xgboost import XGBClassifier as XGBC

battles = pd.read_csv('data/battles.csv')
character_predictions = pd.read_csv('data/character-predictions.csv')
battle, character_pred = q01_feature_engineering(battles,character_predictions)
death_preds = q08_preprocessing(character_pred)
X = death_preds[death_preds.actual == 0].sample(350, random_state = 62).append(death_preds[death_preds.actual == 1].sample(350, random_state = 62)).copy(deep = True).astype(np.float64)
Y = X.actual.values
tX = death_preds[~death_preds.index.isin(X.index)].copy(deep = True).astype(np.float64)
tY = tX.actual.values
X.drop(['SNo', 'actual', 'DateoFdeath'], 1, inplace = True)
tX.drop(['SNo', 'actual', 'DateoFdeath'], 1, inplace = True)

clf_xgb = XGBC(subsample=.8, colsample_bytree=.8, seed=14, max_depth=3)

def q09_XGBoost(X_train,y_train,X_test,y_test,clf_xgb):
    'write your solution here'
    model = clf_xgb
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    pred_prob = clf_xgb.predict_proba(tX)
    roc_auc = roc_auc_score(y_test,pred_prob[:,1])
    accuracy = accuracy_score(y_test,np.argmax(pred_prob, axis =1))
    return roc_auc,accuracy

