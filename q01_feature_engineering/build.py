# %load q01_feature_engineering/build.py
import pandas as pd
import numpy as np

battles = pd.read_csv('data/battles.csv')
character_predictions = pd.read_csv('data/character-predictions.csv')

def q01_feature_engineering(battles,character_predictions):
    'write your solution here'
    battles['defender_count'] = 4 - battles[['defender_1', 'defender_2', 'defender_3', 'defender_4']].isnull().sum(axis = 1)
    battles['attacker_count'] = 4 - battles[['attacker_1','attacker_2','attacker_3','attacker_4']].isnull().sum(axis = 1)
    battles['attacker_comm_count'] = 0
    battles.loc[:,['attacker_comm_count']]=[len(x) if type(x) == list else np.nan for x in battles.attacker_commander.str.split(',')]
    character_predictions['char_in_no_of_books'] = character_predictions[['book1','book2','book3','book4','book5']].sum(axis=1)
    return battles,character_predictions


