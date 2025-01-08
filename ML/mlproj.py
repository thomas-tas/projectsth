import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression

# Set the working directory to the script's directory
script_dir = os.path.dirname(os.path.abspath("winequality-white.csv"))
os.chdir(script_dir)

#read the csv file as pandas dataframe
df = pd.read_csv('winequality-white.csv',sep=';')

#display the first 5 rows
df.head()

#the summary of the data
df.describe()

#check for null values
df.isna().sum()

#check the column names
df.columns

#check for data types of the attributes
df.dtypes

#we need to change the target attribute to 3 classes
#Function for classes
def classify_value(value):
    if value <=4:
        return 0
    elif 4 < value <= 6:
        return 1
    else:
        return 2
# Apply function to create classes in target variable
df['quality'] = df['quality'].apply(classify_value)

df.head()

#display the percentage of the classes
percentages_target=df['quality'].value_counts(normalize=True)*100
percentages_target.plot(kind='bar', color='lightblue')
plt.title('Quality Classes Percentages')
plt.xlabel('Classes')
plt.ylabel('Percentage')

#separate target variable and features
y=df['quality'].values.reshape(-1,1)
x = df.drop(columns=['quality']).values


test_scores={'SVC':[],'Decision Tree':[],'Logistic Regression':[]}
#split the dataset ten times and train the models
for i in range(10):
# split training, validation and test set
    x_tr, x_ts, y_tr, y_ts = train_test_split(x, y, test_size=0.2)  # get training and testing sets
    x_tr, x_vl, y_tr, y_vl = train_test_split(x_tr, y_tr, test_size=0.1)  # get training and validation sets
#standarize the features
    scaler = StandardScaler()
    x_tr = scaler.fit_transform(x_tr)
    x_vl = scaler.transform(x_vl)
    x_ts = scaler.transform(x_ts)

#different hyperparameters for tuning the SVC model
    grid_svc = {
        'C': [0.01, 0.1, 1, 10], # Regularization parameter
        'kernel': ['linear', 'rbf'], # Kernel type
        }
    scores=[] #initialize f1-scores list
#iterate for each hyperparameter
    for j in grid_svc['C']:
        for k in grid_svc['kernel']:

            model1 = SVC(C=j, kernel=k, decision_function_shape='ovo')
            model1.fit(x_tr, y_tr.ravel())
            y_pred1 = model1.predict(x_vl)
            f1 = f1_score(y_vl, y_pred1, average='micro')
            scores.append(f1)

#find the best hyperparameters
            if f1 == max(scores):
                best_param_svc = [j, k]

    # train the model with best hyperparameters
    model1 = SVC(C=best_param_svc[0], kernel=best_param_svc[1], decision_function_shape='ovo')
    model1.fit(x_tr, y_tr.ravel())
    # test the model
    y_pred1 = model1.predict(x_ts)
    f1 = f1_score(y_ts, y_pred1, average='micro')
    test_scores['SVC'].append(f1)

#different hyperparameters for tuning the SVC model
    grid_dtrees = {
        'max_depth': [None, 5, 10, 50],
        'min_samples_split': [2,10,30],
        'min_samples_leaf': [1,5,20]
        }
    scores=[] #initialize f1-scores list for decision trees

#iterate for each hyperparameter
    for j in grid_dtrees['max_depth']:
        for k in grid_dtrees['min_samples_split']:
            for l in grid_dtrees['min_samples_leaf']:
                model2 = DecisionTreeClassifier(max_depth=j,min_samples_split=k,min_samples_leaf=l)
                model2.fit(x_tr, y_tr.ravel())
                y_pred2 = model2.predict(x_vl)
                f1 = f1_score(y_vl,y_pred2,average='micro')
                scores.append(f1)
                #find the best hyperparameters
                if f1==max(scores):
                    best_param_dtrees = [j, k, l]

#train the model with best hyperparameters
    model2 = DecisionTreeClassifier(max_depth=best_param_dtrees[0],min_samples_split=best_param_dtrees[1], min_samples_leaf=best_param_dtr
    model2.fit(x_tr, y_tr.ravel())
    #test the model
    y_pred2 = model2.predict(x_ts)
    f1 = f1_score(y_ts,y_pred2,average='micro')
    test_scores['Decision Tree'].append(f1)

#different hyperparameters for tuning the logistic regression
    # different hyperparameters for tuning the logistic regression
    grid_lr = {
        'C': [0.01, 0.1, 1, 10],  # Regularization parameter
        }
    scores = []  # initialize f1-scores list for logistic regression

#iterate for each hyperparameter
    for j in grid_lr['C']:
        model3 = LogisticRegression(C=j)
        model3.fit(x_tr, y_tr.ravel())
        y_pred3 = model3.predict(x_vl)
        f1 = f1_score(y_vl, y_pred3, average='micro')
        scores.append(f1)
        # find the best hyperparameters
        if f1 == max(scores):
            best_param_dtrees = j
    # train the model with best hyperparameters
    model3 = LogisticRegression(C=best_param_dtrees)
    model3.fit(x_tr, y_tr.ravel())
    # test the model
    y_pred3 = model3.predict(x_ts)
    f1 = f1_score(y_ts, y_pred3, average='micro')
    test_scores['Logistic Regression'].append(f1)

import statistics
#calculate means and standard deviations for each model F1 score
mean_test_scores = {key: round(sum(values) / len(values),4) for key, values in test_scores.items()}
std_test_scores = {key: round(statistics.stdev(values),4) for key, values in test_scores.items()}
for i in mean_test_scores.keys():
print(i," Mean :",mean_test_scores[i])
print(i," Standard Deviation :",std_test_scores[i])

labels = [i for i in mean_test_scores.keys()]

# Define labels and metrics
labels = list(mean_test_scores.keys()) # Models Name
mean = {'F1 score': list(mean_test_scores.values())}
stdeviations = {'F1 score': list(std_test_scores.values())}
strmetrics = ['F1 score']

# Define the width of each bar and adjust spacing
bar_width = 0.6 # Slightly wider for closer bars
x_pos = np.arange(len(labels))

# Create a figure and axis object
fig, ax = plt.subplots(figsize=(12, 6))

# Adjust the subplot layout parameters
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9)

# Set colors and patterns
colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] # Vibrant color palette

# Create a bar plot for the metric
for i, (label, value, std) in enumerate(zip(labels, mean['F1 score'], stdeviations['F1 score'])):
    # Position for each bar
    pos = x_pos[i]
    # Create the bar
    bar = ax.bar(pos, value, color=colors[i], edgecolor='black', yerr=std,
    hatch=None, width=bar_width)
    # Add bar labels
    ax.bar_label(bar, padding=3, fontsize=12)

# Set the x-axis labels and tick positions
ax.set_xticks(x_pos)
ax.set_xticklabels(labels, fontsize=14)

# Add axis labels
ax.set_xlabel('Models', fontsize=16)
ax.set_ylabel(strmetrics[0], fontsize=16)

# Set the y-axis view limit
ax.set_ylim(0, 1.1)
ax.yaxis.set_tick_params(labelsize=12)

# Show the plot
plt.show()

