import matplotlib as plt
import pandas as pd
import numpy as np
import matplotlib

df = pd.read_csv('input/train.csv')

# Remove null Ref values:
df = df[df.loc[:,'Ref'] > 0]
df = df[df.loc[:,'Ref':'Kdp_5x5_90th'].count(1) > 0]
