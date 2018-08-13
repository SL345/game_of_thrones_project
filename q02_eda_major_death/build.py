# %load q02_eda_major_death/build.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys,os
sys.path.append(os.path.join(os.path.dirname(os.curdir)))
from greyatomlib.game_of_thrones.q01_feature_engineering.build import q01_feature_engineering

battles = pd.read_csv('data/battles.csv')
character_predictions = pd.read_csv('data/character-predictions.csv')

def q02_eda_major_death(battles):
    'write your solution here'
    battles.groupby('year').sum()[['major_death','major_capture']].plot.bar()




