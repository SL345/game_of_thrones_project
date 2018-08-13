# %load q06_battles_on_each_region/build.py
import pandas as pd
import numpy as np
import seaborn as sns
sns.set_style('white')
import matplotlib.pyplot as plt
import sys,os
sys.path.append(os.path.join(os.path.dirname(os.curdir)))
from greyatomlib.game_of_thrones.q01_feature_engineering.build import q01_feature_engineering
plt.switch_backend('agg') 

battles = pd.read_csv('data/battles.csv')
character_predictions = pd.read_csv('data/character-predictions.csv')

def q06_battles_on_each_region(battles):
    'write your solution here'
    battles1,character_predictions1 = q01_feature_engineering(battles,character_predictions)
    df = battles1.groupby('region').sum()[['major_death', 'major_capture']]

    df1 = battles1.region.value_counts()
    df1 = df1.to_frame()
    df1 = df1.sort_index(ascending = False)
    df1.columns = ['No.of battles']

    df2 = pd.concat([df,df1],axis=1)

    c = df2.plot.bar()
    c.set(xlabel='region',ylabel='No.of events')
    return df


