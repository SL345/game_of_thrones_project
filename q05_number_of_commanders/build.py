# %load q05_number_of_commanders/build.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
import sys,os
sys.path.append(os.path.join(os.path.dirname(os.curdir)))
from greyatomlib.game_of_thrones.q01_feature_engineering.build import q01_feature_engineering
plt.switch_backend('agg') 

battles = pd.read_csv('data/battles.csv')
character_predictions = pd.read_csv('data/character-predictions.csv')

def q05_number_of_commanders(battles):
    'write your solution here'
    battles,character_predictions = q01_feature_engineering(battles,character_predictions)

    sns.boxplot(x = 'att_comm_count', y ='attacker_king',data=battles)
    plt.xlabel('No.of Attackers')
    plt.ylabel('Attacker King')
    plt.show()



