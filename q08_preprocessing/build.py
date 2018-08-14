# %load q08_preprocessing/build.py
import pandas as pd
import numpy as np
import sys,os
sys.path.append(os.path.join(os.path.dirname(os.curdir)))
from greyatomlib.game_of_thrones.q01_feature_engineering.build import q01_feature_engineering
from greyatomlib.game_of_thrones.q07_culture_survival.build import q07_culture_survival

battles = pd.read_csv('data/battles.csv')
character_predictions = pd.read_csv('data/character-predictions.csv')



def q08_preprocessing(character_predictions):
    'write your solution here'
    copy_df = character_predictions.copy(deep=True)

    copy_df.loc[:,['culture']] = [q07_culture_survival(x) for x in copy_df.culture.fillna('')]
    copy_df['title'] =pd.factorize(copy_df['title'])[0]
    copy_df['culture'] =pd.factorize(copy_df['culture'])[0]
    copy_df['mother'] =pd.factorize(copy_df['mother'])[0]
    copy_df['father'] =pd.factorize(copy_df['father'])[0]
    copy_df['house'] =pd.factorize(copy_df['house'])[0]
    copy_df['heir'] =pd.factorize(copy_df['heir'])[0]
    copy_df['spouse'] =pd.factorize(copy_df['spouse'])[0]
    copy_df.drop(['name', 'alive', 'pred', 'plod', 'isAlive', 'dateOfBirth'],1,inplace=True)
    copy_df.replace('.',' ',inplace=True)
    copy_df.replace('_',' ',inplace=True)
    copy_df.fillna(-1,inplace=True)
    return copy_df



