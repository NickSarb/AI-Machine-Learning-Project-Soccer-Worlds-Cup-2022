#1. Import libraries
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


#2. Explore and visualize the data to gain insights.

#2.1 Let's look at correlations with regard to our target
corr_matrix = stats.corr(numeric_only=True)
#Compare correlations based on our target (possession)
corr_matrix["Poss"].sort_values(ascending=False)

#2.2 Plot Poss vs. PrgP using sns.lineplot, and create a scatterplot for Poss and PrgP
g = sns.lineplot(x="PrgP", y="Poss", data=stats, errorbar=None)

# Create a scatterplot of Poss and PrgP
X = stats["PrgP"]
y = stats["Poss"]
# Plot points
fig, pl = plt.subplots()
pl.scatter(X, y, color = 'b')
plt.xlabel("PrgP")
plt.ylabel("Poss")

#2.3 Plot Poss vs. Expected Goal(Eg) and Goal(Gls) using sns.lineplot, and create a scatterplot for Plot Poss vs. Expected Goal(Eg) and Goal(Gls)
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

#2.4 Plot Poss vs. Assist(Ast) and Goal(Gls) using sns.lineplot, and create a scatterplot for Plot Poss vs. Assist(Ast) and Goal(Gls)
plt.figure(figsize=(8, 6))

# Scatter plot for Possession vs Age
sns.scatterplot(x='Poss', y='Ast', data=stats, s=100, color='skyblue', edgecolor='black', alpha=0.7, label='Assists')

# Scatter plot for Goals
sns.scatterplot(x='Poss', y='Gls', data=stats, s=125, color='red', edgecolor='black', marker='o', label='Goals')

# Add title and labels
plt.title('Possession vs. Goals or Assists')
plt.xlabel('Possession (%)')
plt.ylabel('Age')

# Add grid lines
plt.grid(True, linestyle='--', alpha=0.5)

# Add trend line
sns.regplot(x='Poss', y='Ast', data=stats, scatter=False, color='red', label='Trend of Assists')
sns.regplot(x='Poss', y='Gls', data=stats, scatter=False, color='blue', label='Trend of Goals')

# Show the plot
plt.legend()
plt.tight_layout()
plt.show()

#2.5 A histograph to show distribution of Possession(Poss)
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


#3. Prepare the data for Machine Learning Algorithms

#3.1 Remove unwanted expected data columns
# Dropping expected coloumns: xG, npxG, xAG, npxG+xAG, xG90, xAG90, xG+xAG90,npxG90, npxG+xAG90
# They are coloumns that do not effect the prediction and are not needed
stats.drop(labels=['xG', 'npxG', 'xAG', 'npxG+xAG', 'xG90', 'xAG90', 'xG+xAG90', 'npxG90', 'npxG+xAG90'], axis=1, inplace=True)
# Confirm expected coloumns have been removed
stats.columns

#3.2 Create a pipeline that will
#Scale the numerical columns using StandardScaler. Do not scale the target
#Encode the categorical columns using OneHotEncoder
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
                                    remainder='passthrough')

#3.3 Displaying the pipeline
# Show the pipeline
preprocessing

# Apply the preprocessing pipeline on the dataset
stats_prepared = preprocessing.fit_transform(stats)

# Scikit-learn strips the column headers in most cases, so just add them back on afterward.
feature_names=preprocessing.get_feature_names_out()
stats_prepared = pd.DataFrame(data=stats_prepared, columns=feature_names)

stats_prepared


#4. Select a model and train it

#4.1 Split the dataset into a training dataset (80%) and testing dataset.
from sklearn.model_selection import train_test_split

X = stats_prepared.drop(["remainder__Poss"], axis=1)
y = stats_prepared["remainder__Poss"]

#Split dataset into training set(0.8), testing set(0.2) with random state(42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

#4.2 Train a Linear Regression model
from sklearn.linear_model import LinearRegression

lr_model = LinearRegression()

#Fit our training set into a linear regression model
lr_model.fit(X_train,y_train)

#Test your model on the test set, and report on the Mean Squared Error
# Predict the outcome of test data and output the mean squared error
lr_y_predict = lr_model.predict(X_test)

from sklearn.metrics import mean_squared_error as mse
lr_mse=mse(y_test, lr_y_predict)
lr_mse

#4.3 Train a Linear Regression model using KFold cross-validation with 5 folds, and report on the cross validation score, use negative mean squared error as the cross validation metric.
from sklearn.model_selection import cross_val_score, KFold

# Train the model using KFold cross-validation with 5 folds
scores = cross_val_score(lr_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
scores

#4.4 Calculate the mean of the cross-validation scores to get an overall assessment of the model's performance.
mean_score = -scores.mean()  # Take the negative value to get the mean squared error

print(f'Cross-Validation Mean Score: {mean_score}')

#4.5 Train a Linear Regression model using Ridge and Lasso regularization with alpha=1
from sklearn.linear_model import LinearRegression, Ridge, Lasso

#Train data using Ride Regression
RidgeRegression = Ridge(alpha=1)
ridge_model = RidgeRegression.fit(X_train, y_train)

#Train data using Lasso Regression
LassoRegression = Lasso(alpha=1)
lasso_model = LassoRegression.fit(X_train, y_train)

#Train and Test Model Elastic Net
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error

#Train data using Elastic Net
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic_net.fit(X_train, y_train)
y_pred = elastic_net.predict(X_test)

# Evaluating the model
en_mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', en_mse)

#4.6 Train and test Polynomial Regression model
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
degree = 4  # Degree of polynomial features
poly_features = PolynomialFeatures(degree=degree)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)
poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)

# Making predictions on the test set
y_pred = poly_model.predict(X_test_poly)

# Evaluating the model
polyR_mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', polyR_mse)

#4.7 Test your models on the test dataset and report on the MSE
#Testing various models on test dataset
Ridge_y_predict = ridge_model.predict(X_test)
Lasso_y_predict = lasso_model.predict(X_test)
en_y_pred = elastic_net.predict(X_test)
poly_y_pred = poly_model.predict(X_test_poly)

#Reporting on the MSE of each test data
en_mse = mean_squared_error(y_test, en_y_pred)
polyR_mse = mean_squared_error(y_test, poly_y_pred)
ridge_mse = mean_squared_error(y_test, Ridge_y_predict)
lasso_mse=mean_squared_error(y_test, Lasso_y_predict)

#4.8 Cross Validation for all models
# Train the model using KFold cross-validation with 5 folds
scores_ridge = cross_val_score(ridge_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
scores_elastic = cross_val_score(elastic_net, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
scores_lasso = cross_val_score(lasso_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
scores_poly = cross_val_score(poly_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')

# Take the negative value to get the mean squared error
mean_score_ridge = -scores_ridge.mean()
mean_score_lasso = -scores_lasso.mean()
mean_score_poly = -scores_poly.mean()
mean_score_elastic = -scores_elastic.mean()

# Print and compare all models
print(f'Cross-Validation Linear Regression Mean Score: {mean_score}')
print(f'Cross-Validation for Elastic Net Regression Mean Score: {mean_score_elastic}')
print(f'Cross-Validation for Ridge Mean Regression Score: {mean_score_ridge}')
print(f'Cross-Validation for Lasso Mean Regression Score: {mean_score_lasso}')
print(f'Cross-Validation for Polynomial Regression Mean Score: {mean_score_poly}')

#Compare results from all models
# Print and compare all models
print(f'Linear Regression MSE: {lr_mse}')
print(f'Ridge Regression MSE: {ridge_mse}')
print(f'Lasso Regression MSE: {lasso_mse}')
print(f'Elastic Net MSE: {en_mse}')
print(f'Polynomial Regression MSE: {polyR_mse}')

#4.9 Plot Comparison of Cross-Validation Mean Scores and Mean Square Error for Different Regression Models
# Data
models = ['Linear', 'Elastic Net', 'Ridge', 'Lasso', 'Polynomial']
mean_scores = [mean_score, mean_score_elastic, mean_score_ridge, mean_score_lasso, mean_score_poly]
mses = [lr_mse, en_mse, ridge_mse, lasso_mse, polyR_mse]

# Plotting
fig, ax = plt.subplots(figsize=(10, 8))

# Bar width
bar_width = 0.35
index = np.arange(len(models))

# Plot mean scores
ax.bar(index - bar_width/2, mean_scores, bar_width, label='Mean Score', color='skyblue')

# Plot MSEs
ax.bar(index + bar_width/2, mses, bar_width, label='Mean Square Error', color='salmon')

# Add labels, title, legend, and grid
ax.set_xlabel('Models')
ax.set_ylabel('Scores / MSE')
ax.set_title('Comparison of Cross-Validation Mean Scores and Mean Square Error for Different Regression Models')
ax.set_xticks(index)
ax.set_xticklabels(models)
ax.legend()
ax.grid(axis='y')

plt.tight_layout()
plt.show()

#4.10 Plot Ridge Regression Coefficients vs Lasso Regression Coefficients
ridge_coefficients = ridge_model.coef_
lasso_coefficients = lasso_model.coef_

# Plot coefficients
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.bar(range(len(ridge_coefficients)), ridge_coefficients)
plt.title('Ridge Regression Coefficients')
plt.xlabel('Feature Index')
plt.ylabel('Coefficient Value')

plt.subplot(1, 2, 2)
plt.bar(range(len(lasso_coefficients)), lasso_coefficients)
plt.title('Lasso Regression Coefficients')
plt.xlabel('Feature Index')
plt.ylabel('Coefficient Value')

plt.tight_layout()
plt.show()


#5. Graphs for best AI Algorithm

#5.1 Plot Actual data vs Predicted data
# Plotting the predicted values versus the actual values for training data
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(y_train, lr_model.predict(X_train), color='blue')
plt.plot([0, max(y_train)], [0, max(y_train)], color='red', linestyle='--')
plt.title('Training Data: Actual vs Predicted')
plt.xlabel('Actual')
plt.ylabel('Predicted')

# Plotting the predicted values versus the actual values for testing data
plt.subplot(1, 2, 2)
plt.scatter(y_test, lr_y_predict, color='green')
plt.plot([0, max(y_test)], [0, max(y_test)], color='red', linestyle='--')
plt.title('Testing Data: Actual vs Predicted')
plt.xlabel('Actual')
plt.ylabel('Predicted')

plt.tight_layout()
plt.show()

#5.2 Plot accuracy comparison between training and testing data
# Calculate the mean squared error for training and testing data
train_mse = mse(y_train, lr_model.predict(X_train))
test_mse = mse(y_test, lr_y_predict)

# Calculate the accuracy in percentage
train_accuracy = 100 * (1 - (train_mse / np.var(y_train)))
test_accuracy = 100 * (1 - (test_mse / np.var(y_test)))

# Plotting the accuracy comparison
plt.figure(figsize=(8, 5))
plt.bar(['Training Data', 'Testing Data'], [train_accuracy, test_accuracy], color=['blue', 'green'])
plt.title('Accuracy Comparison between Training and Testing Data')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 100)  # Set the y-axis limit from 0 to 100
plt.show()

#5.3 Plot residuals against predicted values
# Calculating residuals
residuals = y_test - lr_y_predict

plt.figure(figsize=(10, 6))

# Plotting residuals against predicted values
plt.scatter(lr_y_predict, residuals, color='blue')
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Residual Plot')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.grid(True)

# Calculate mean absolute error
mae = np.mean(np.abs(residuals))

# Add MAE to the plot
plt.text(0.1, 0.9, f'Mean Average Error: {mae:.2f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

plt.show()

# References:
# end-to-end-studentsPerformance-RA-eclass-W24
# svm-Classification-Iris-Moons-Wine-RA-eclass-W24
# EECS3401 Lecture Material








