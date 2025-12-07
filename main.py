import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from country import Country

#RESTRICTIONS
"""
No two of the same conference (except UEFA)
Teams ranked 1, 2, then 3, 4 must be in opposite brackets
ABCJKL,DEFGHI
Mex A1
Can B1
USA D1
Each group requires 1 UEFA
"""

teams = 'teams.csv'

teams_df = pd.read_csv(teams)
print(teams_df.head())

