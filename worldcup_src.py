# Import libraries
import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Dataset is being read from our github page at https://raw.githubusercontent.com/chiemgiabaost/Project_group_12/main/2022worldcup.csv
url = "https://raw.githubusercontent.com/chiemgiabaost/Project_group_12/main/2022worldcup.csv"
stats = pd.read_csv(url, sep=',')

#Created a backup copy of the dataset
stats_backup = stats

#Compare correlations based on our target (possession)
corr_matrix["Poss"].sort_values(ascending=False)
