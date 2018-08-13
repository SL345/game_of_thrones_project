# %load q04_pairs_fought_the_most_battles/build.py
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import sys,os
sys.path.append(os.path.join(os.path.dirname(os.curdir)))
from greyatomlib.game_of_thrones.q01_feature_engineering.build import q01_feature_engineering
plt.switch_backend('agg') 

battles = pd.read_csv('data/battles.csv')
character_predictions = pd.read_csv('data/character-predictions.csv')

def q04_pairs_fought_the_most_battles(battles):
    'write your solution here'
    # Ignoring records where either attacker_king or defender_king is null. Also ignoring one record where both have the same value.
    lst = list(Counter([tuple(set(x)) for x in battles.dropna(subset = ['attacker_king', 'defender_king'])[['attacker_king', 'defender_king']].values if len(set(x)) > 1]).items())
    df = pd.DataFrame(lst,columns = ['Attacker King vs Defender King','No. of Battles'])
    sns.barplot(x='No. of Battles', y='Attacker King vs Defender King', data=df,
            label='No. of Battles', color='b')
    plt.show()
    return lst


