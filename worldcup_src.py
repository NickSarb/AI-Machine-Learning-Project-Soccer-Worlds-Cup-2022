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

#plot Poss vs PrgP
g = sns.lineplot(x="PrgP", y="Poss", data=stats, errorbar=None)

# Create a scatterplot of Poss and PrgP
X = stats["PrgP"]
y = stats["Poss"]
# Plot points
fig, pl = plt.subplots()
pl.scatter(X, y, color = 'b')
plt.xlabel("PrgP")
plt.ylabel("Poss")

# Plot Poss vs. Assist(Ast) and Goal(Gls) using sns.lineplot, and create a scatterplot for Plot Poss vs. Assist(Ast) and Goal(Gls)
plt.figure(figsize=(8, 6))

sns.scatterplot(x='Poss', y='xG', data=stats, s=100, color='skyblue', edgecolor='black', alpha=0.7, label='Expected Goals')

# Scatter plot for Goals
sns.scatterplot(x='Poss', y='Gls', data=stats, s=125, color='red', edgecolor='black', label='Goals')

# Add title and labels
plt.title('Possession vs. Goals or Expected Goals(EG)')
plt.xlabel('Possession (%)')
plt.ylabel('Goals')

# Add grid lines
plt.grid(True, linestyle='--', alpha=0.5)

# Add trend line
sns.regplot(x='Poss', y='xG', data=stats, scatter=False, color='blue', label='Trend of EG')
sns.regplot(x='Poss', y='Gls', data=stats, scatter=False, color='red', label='Trend of Goals')

# Show the plot
plt.legend()
plt.tight_layout()
plt.show()

# A histograph to show distribution of Possession(Poss)
plt.figure(figsize=(8, 6))

# Histogram for Possession
sns.histplot(stats['Poss'], bins=10, color='skyblue', edgecolor='black')

# Add title and labels
plt.title('Distribution of Possession')
plt.xlabel('Possession (%)')
plt.ylabel('Frequency')

# Add grid lines
plt.grid(True, linestyle='--', alpha=0.5)

# Show the plot
plt.tight_layout()
plt.show()

# Dropping expected coloumns: xG, npxG, xAG, npxG+xAG, xG90, xAG90, xG+xAG90,npxG90, npxG+xAG90
# They are coloumns that do not effect the prediction and are not needed
stats.drop(labels=['xG', 'npxG', 'xAG', 'npxG+xAG', 'xG90', 'xAG90', 'xG+xAG90', 'npxG90', 'npxG+xAG90'], axis=1, inplace=True)

# Confirm expected coloumns have been removed
stats.columns

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

#create the cat and num columns
num_cols = stats.select_dtypes(include='number').columns.to_list()
cat_cols = stats.select_dtypes(exclude='number').columns.to_list()


#exclude the target from numerical columns
num_cols.remove("Poss")

#create pipelines for numeric and categorical columns
num_pipeline = make_pipeline(StandardScaler())
cat_pipeline = make_pipeline(OneHotEncoder())

#use ColumnTransformer to set the estimators and transformations

preprocessing = ColumnTransformer([('num', num_pipeline, num_cols),
                                   ('cat', cat_pipeline, cat_cols)],
                                    remainder='passthrough'
                                 )

# Show the pipeline
preprocessing

# Apply the preprocessing pipeline on the dataset
stats_prepared = preprocessing.fit_transform(stats)

# Scikit-learn strips the column headers in most cases, so just add them back on afterward.
feature_names=preprocessing.get_feature_names_out()
stats_prepared = pd.DataFrame(data=stats_prepared, columns=feature_names)

stats_prepared
