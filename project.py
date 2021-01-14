import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, r2_score, silhouette_score, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from datetime import timedelta, datetime
import seaborn as sns 
import statistics
import re
import copy

df = pd.read_csv("latestdata.csv")

# 1.Analysis of the dataset
print("=== 1.Analysis of the dataset === ")

# A

# Keep usefull columns
#df = df.loc[:, [['age', 'sex', 'outcome', 'chronic_disease_binary', 'date_onset_symptoms', 'date_admission_hospital', 'date_confirmation', 'lives_in_Wuhan', 'travel_history_location', 'outcome', 'date_death_or_discharge']]
df = df.drop(columns=['ID','additional_information', 'reported_market_exposure', 'source', 'sequence_available', 'admin3', 'admin2', 'admin1', 'admin_id', 'data_moderator_initials', 'notes_for_discussion', 'travel_history_binary'])

df = df.dropna(subset=['sex', 'latitude', 'longitude'])

def age(x):
    x_str = str(x)
    if "-" in x_str:
        ages = x_str.split("-")
        if not '' in ages:
            ages = [int(i) for i in ages]
            # maybe floor
            return statistics.mean(ages)
        else:
            return ages[0]
    elif "+" in x_str:
        ages = x_str.split("+")
        return int(ages[0])
    elif "month" in x_str:
        return 1
    else:
        return x

df.age = df.age.apply(age)

## Too slow
# Clean of the dataset
# for index, row in df.iterrows():
#     age = str(row['age'])
#     if "-" in age:
#         ages = age.split("-")
#         if not '' in ages:
#             ages = [int(i) for i in ages]
#             # maybe floor
#             df.loc[index, 'age'] = statistics.mean(ages)
#         else:
#             df.loc[index, 'age'] = ages[0]
            # if re.match("[0-9]{1,3}\-", ages[0]):
            #     df.loc[index, 'age'] = ages[0]
            # elif re.match("-[0-9]{1,3}", ages[1]):
            #     df.loc[index, 'age'] = ages[1]
    # elif "+" in age:
    #     ages = age.split("+")
    #     df.loc[index, 'age'] = ages[0]
    # date_onset_symptoms = str(row['date_onset_symptoms'])
    # if "-" in date_onset_symptoms:
    # date_admission_hospital = str(row['date_admission_hospital'])
    # if "-" in date_admission_hospital:
    # date_confirmation = str(row['date_confirmation'])
    # if "-" in date_confirmation:
    

# Fill missing values
df["age"] = pd.to_numeric(df["age"])
# df["date_onset_symptoms"] = pd.to_datetime(df["date_onset_symptoms"])
# df["date_admission_hospital"] = pd.to_datetime(df["date_admission_hospital"])
# df["date_confirmation"] = pd.to_datetime(df["date_confirmation"])

values = {'age': df["age"].mean(), 'outcome': 'unknown', 'lives_in_Wuhan': "no"}
df = df.fillna(value=values)

# New column : sex_cat : 1 if male 0 if female
df = df.assign(sex_cat = (df['sex'] == 'male').astype(int))

# New column : is_death : 1 if 'outcome' contains death 0 if no
df = df.assign(is_death = (df['outcome'].str.contains('death|died|deceased|dead', regex=True, case=False, na=0)).astype(int))

df = df.assign(death_or_discharge = (df['date_death_or_discharge'].isnull() == False).astype(int))

df = df.assign(visited_Wuhan = df['travel_history_location'].str.contains('wuhan', case=False, na=0))

df = df.assign(visited_Wuhan = ((df["lives_in_Wuhan"] == "yes") | (df['visited_Wuhan'] == 1)).astype(bool))

df = df.assign(lives_in_Wuhan = (df["lives_in_Wuhan"] == "yes").astype(bool))

# 0-14 : 0
# 15-44 : 1
# 45-64 : 2
# 65-74 : 3
# +75 : 4
# def age_cat(df_age):
#     for index, row in df.iterrows():
#         age = row['age']
#         if age <= 14:
#             cat = 0
#         elif age <= 44:
#             cat = 1
#         elif age <= 64:
#             cat = 2
#         elif age <= 75:
#             cat = 3
#         else:
#             cat = 4
#         df_age.loc[index] = cat
#     return df_age

# df = df.assign(age_cat = lambda x: age_cat(x['age']))

df_types = df[['age', 'longitude', 'latitude', 'is_death', 'chronic_disease_binary', 'sex_cat', 'death_or_discharge', 'visited_Wuhan', 'lives_in_Wuhan']]

print("number of rows")
print(df.size)

print("first 5 rows")
print(df.head())

print("dtypes of data")
print(df.dtypes)

print("correlations between the variables")
df_corr = df_types.corr().round(1)
sns.heatmap(data=df_corr, annot=True)

print("number of null data for each column")
print(df.isnull().sum())
# B

#PCA
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df[['age', 'latitude', 'longitude', 'chronic_disease_binary', 'lives_in_Wuhan', 'visited_Wuhan', 'sex_cat', 'death_or_discharge', 'is_death']])

targets = ['discharged', 'died']
colors = ['g', 'r']

fig = plt.figure()
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
for target, color in zip(targets,colors):
    goal = 1 if target=='died' else 0
    indicesToKeep = df_types['is_death'] == goal
    ax.scatter(df_pca[indicesToKeep, 0],
            df_pca[indicesToKeep, 1],
            c = color,
            s = 10)
ax.legend(targets)
ax.grid()

# fig = plt.figure()
# ax = fig.add_subplot(1,1,1) 
# ax.set_xlabel('Principal Component 1', fontsize = 15)
# ax.set_ylabel('Principal Component 2', fontsize = 15)
# ax.set_title('2 component PCA', fontsize = 20)
# ax.scatter(df_pca[:, 0],
#             df_pca[:, 1],
#             s = 50)
# ax.grid()

#plt.show()

# 2.Bayes Nets
print("=== 2. Bayes Nets === ")

# A. Probability for a person to have symptoms of COVID-19 if this person visited Wuhan
# Probability of visit Wuhan
# nb_wuhan = df['travel_history_location'].str.contains('wuhan', case=False, na=0).sum(axis=0)
# total_nb = df['travel_history_location'].shape[0]

# p_wuhan = nb_wuhan/total_nb

# df_symp = df.assign(symptoms = (df['symptoms'] != "").astype(int))
# nb_symptoms = df_symp['symptoms'].sum(axis=0)
# total_nb = df['symptoms'].shape[0]

# p_symptoms = nb_symptoms/total_nb

# df_symp = df_symp.loc[df_symp['symptoms'] == 1]
# nb_wuhan = df_symp['travel_history_location'].str.contains('wuhan', case=False, na=0).sum(axis=0)
# total_nb = df_symp['travel_history_location'].shape[0]
# # P(wuhan|symptoms)
# p_wuhan_symptoms = nb_wuhan/total_nb

# # P(wuhan|symptoms) Bayes
# print(p_wuhan_symptoms*p_symptoms/p_wuhan)

df_wuhan = df.loc[df['visited_Wuhan'] == 1]
df_symptoms = df_wuhan.loc[df_wuhan['symptoms'].isnull() == False]

nb_symptoms = df_symptoms.shape[0]
nb_total = df_wuhan.shape[0]

# P(symptoms|wuhan)
prob_symptoms = nb_symptoms/nb_total
print("P(symptoms|Wuhan) = {}".format(prob_symptoms))

# B
# Probability for a person to be true patient if this person has symptoms of COVID-19 and visited Wuhan
df_patient = df_symptoms.loc[df_symptoms['date_admission_hospital'].isnull() == False]

nb_patient = df_patient.shape[0]
nb_total = nb_symptoms

# P(patient|symptoms+Wuhan)
prob_patient = nb_patient/nb_total
print("P(patient|symptoms+Wuhan) = {}".format(prob_patient))

# C
# Propability for a person to death if this person visited Wuhan
df_death_wuhan = df_wuhan.loc[df_wuhan["date_death_or_discharge"].isnull() == False]

nb_total = df_wuhan.shape[0]
nb_death = df_death_wuhan.shape[0]

# P(death|Wuhan)
prob_death = nb_death/nb_total
print("P(death|Wuhan) = {}".format(prob_death))

#P(symptoms|Wuhan) = 0.42718446601941745
#P(patient|symptoms+Wuhan) = 0.8409090909090909
#P(death|Wuhan) = 0.2524271844660194

# D. Average recovery interval for a patient if this person visited Wuhan
df_recovery = df.dropna(subset=["date_onset_symptoms","date_death_or_discharge"])
df_recovery = df_recovery[(df_recovery["visited_Wuhan"] == True) & (df_recovery["is_death"] == 1)]

def date_handler(x):
    x_str = str(x)
    if "-" in x_str:
        dates = x_str.split("-")
        return dates[0]
    else:
        return x

df_recovery.date_onset_symptoms = df_recovery.date_onset_symptoms.apply(date_handler)

recovery_array = []

for row in df_recovery.T.to_dict().values():
    start_date = datetime.strptime(row["date_onset_symptoms"],"%d.%m.%Y").strftime("%s")
    end_date = datetime.strptime(row["date_death_or_discharge"],"%d.%m.%Y").strftime("%s")
    recovery_time = int(end_date) - int(start_date) 
    recovery_array.append(recovery_time)

df_recovery["recovery_time"] = recovery_array

recovery_seconds = int(df_recovery["recovery_time"].mean())

average_recovery = timedelta(seconds=recovery_seconds) 
print("Average recovery time")
print(average_recovery)

# 3.Machine Learning
print("=== 3. Machine Learning === ")

df_ml = df[['age', 'latitude', 'longitude', 'chronic_disease_binary', 'lives_in_Wuhan', 'visited_Wuhan', 'sex_cat', 'death_or_discharge', 'is_death']]

# KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)

#print(df.outcome.unique())

X = df_ml.drop(columns=['is_death']).values
y = df_ml['is_death'].values

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3)

# create the model
neigh.fit(X_train, y_train)

# test the model
y_pred = neigh.predict(X_test)

# A

# confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

tn, fp, fn, tp = conf_matrix.ravel()

# Accuracy
accuracy = (tp+tn)/(tp+tn+fp+fn)
print("Accuracy = {}".format(accuracy))

# Recall
recall = tp/(tp+fn)
print("Recall = {}".format(recall))

# Specifity
specifity = tn/(tn+fp)
print("Specifity = {}".format(specifity))

# Precision
precision = tp/(tp+fp) if (fp > 0 | tp > 0) else 0
print("Precision = {}".format(precision))

# F-Measure
f_measure = 2*precision*recall/(precision+recall)
print("F-Measure = {}".format(f_measure))

# Classification report
print(classification_report(y_test, y_pred, target_names=targets, zero_division=0))

# B
X = df_ml.drop(columns=["age"]).values
y = df_ml["age"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=5)

scaler = StandardScaler().fit(X)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

knn_regressor = KNeighborsRegressor(3)

knn_regressor.fit(X_train, y_train)

y_pred = np.around(knn_regressor.predict(X_test), decimals=0)

print("MSE : {:.2f}".format(np.sqrt(mean_squared_error(y_test, y_pred))))
print("MAD : {:.2f}".format(np.sqrt(mean_absolute_error(y_test, y_pred))))

plt.figure()
plt.scatter(y_test[::200],y_pred[::200], color='coral') 
plt.ylabel('Predicted Age')
plt.xlabel('Actual Age')

# C
X = df_ml.values[::50]

scores = []
for i in range(2,15):
    kmeans = KMeans(i)
    kmeans.fit(X)
    scores.append(silhouette_score(X, kmeans.labels_))

max_score = max(scores)

nb_clusters = [i for i,j in enumerate(scores) if j == max_score][0]

kmeans = KMeans(nb_clusters)
kmeans.fit(X)

print(f"The best silhouette score here is : {max_score} and  represents {nb_clusters}.")

# 4.Improving the results and Theoretical formalism
print("=== 4. Improving the results and Theoretical formalism === ")
# A

X = df_ml.drop(columns=["is_death"]).values
y = df_ml["is_death"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=5)

scaler = StandardScaler().fit(X)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# B

# C

# hyperparameters to set
param_grid = {"n_neighbors": [1,2,3,4,5,6,7,9,10,11,13,15]}

classifier = GridSearchCV( 
    KNeighborsClassifier(),
    param_grid, # hyperparameters to test
    cv=5, # folds for cross validation (5 or 10 generally) scoring=score # score to optimize
)

# optimize the classifier on the training set
classifier.fit(X_train, y_train)

print("Best Hyperparameters on training test")
print(classifier.best_params_)
print("Cross validation results") 

probs = zip(classifier.cv_results_['mean_test_score'], classifier.cv_results_['std_test_score'], classifier.cv_results_['params'])

for mean, std, params in probs:
    print("Accuracy = {:.3f} (+/-{:.03f}) for {}".format(mean, std*2, params))

#plt.show()