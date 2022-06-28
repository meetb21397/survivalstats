import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
from sklearn.impute import SimpleImputer
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score,classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
import xgboost
from sklearn.neighbors import LocalOutlierFactor


data = pd.read_csv('C:/Users/m997t/Downloads/echocardiogram.csv')
data.head()
data = data.drop(['name', 'group', 'aliveat1'], axis=1)
data.head()
data.isnull().sum()

mean = SimpleImputer(missing_values = np.nan, strategy = 'mean')
Columns = ['age', 'pericardialeffusion', 'fractionalshortening', 'epss', 'lvdd', 'wallmotion-score']
X = mean.fit_transform(data[Columns])
df_X = pd.DataFrame(X,
                    columns = Columns)
keep = ['survival', 'alive']
data_keepcolumn = data[keep]
data = pd.concat([data_keepcolumn, df_X], axis = 1)
data = data.dropna()
print(data.isnull().sum())
print(data.shape)

sns.pairplot(data)

data.loc[data.alive == 1, 'dead'] = 0
data.loc[data.alive == 0, 'dead'] = 1
data.groupby('dead').count()

#Kaplan Meier Curve
kmf = KaplanMeierFitter()
X= data['survival']
Y = data['dead']
kmf.fit(X, event_observed = Y)
kmf.plot()
plt.title("Kaplan Meier estimates")
plt.xlabel("Month after heart attack")
plt.ylabel("Survival")
plt.show()


#From the plot we can see that the survival rate decreases with the increase in the number of months.The Kaplan estimate is 1 for the initial days following the heart treatment.It gradually decreases to around 0.05 after 50 months.

print("The median survival time :",kmf.median_survival_time_)

print(kmf.survival_function_)

age_group = data['age'] < statistics.median(data['age'])
ax = plt.subplot(111)
kmf.fit(X[age_group], event_observed = Y[age_group], label = 'below 62')
kmf.plot(ax = ax)
kmf.fit(X[~age_group], event_observed = Y[~age_group], label = 'above 62')
kmf.plot(ax = ax)
plt.title("Kaplan Meier estimates by age group")
plt.xlabel("Month after heart attack")
plt.ylabel("Survival")

score_group = data['wallmotion-score'] < statistics.median(data['wallmotion-score'])
ax = plt.subplot(111)
kmf.fit(X[score_group], event_observed = Y[score_group], label = 'Low score')
kmf.plot(ax = ax)
kmf.fit(X[~score_group], event_observed = Y[~score_group], label = 'High score')
kmf.plot(ax = ax)
plt.title("Kaplan Meier estimates by wallmotion-score group")
plt.xlabel("Month after heart attack")
plt.ylabel("Survival")


features_with_null = [features for features in data.columns if data[features].isnull().sum()>0]
for feature in features_with_null:print(feature, ':', round(data[feature].isnull().mean(), 4), '%')

for feature in features_with_null:print(feature, ':', data[feature].unique())

data = data.dropna(subset=['alive'])
data['alive'].isnull().sum()

discrete_features = ['pericardialeffusion']
continuous_features = data.drop(['pericardialeffusion', 'alive'], 1).columns
label = ['alive']
print(continuous_features)
for feature in discrete_features:
    data[feature] = data[feature].fillna(data[feature].mode()[0])
for feature in continuous_features:
    data.boxplot(feature)
    plt.title(feature)
    plt.show()
#Additional plots of wallmotion index and further more
    features_with_outliers = ['wallmotion-score', 'wallmotion-index', 'mult']
for feature in continuous_features:
     if feature in features_with_outliers:
         data[feature].fillna(data[feature].median(), inplace=True)
     else:
         data[feature].fillna(data[feature].mean(), inplace=True)

         from sklearn.neighbors import LocalOutlierFactor

         lof = LocalOutlierFactor()
         outliers_rows = lof.fit_predict(data)

         mask = outliers_rows != -1
data.isnull().sum()
data = data[mask]

data1 = pd.get_dummies(data, columns = discrete_features, drop_first = True)
scaler = StandardScaler()
data1[continuous_features] = scaler.fit_transform(data1[continuous_features])
data1.head()
X = data1.drop(['alive'], 1)
y = data1['alive']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
X_train.shape, X_test.shape

accuracy = {}

data1.head()
X = data1.drop(['alive'], 1)
y = data1['alive']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
X_train.shape, X_test.shape

accuracy = {}

model1 = LogisticRegression(max_iter=200)
model1.fit(X_train, y_train)
y_pred1 = model1.predict(X_test)
print(accuracy_score(y_test, y_pred1))
accuracy[str(model1)] = accuracy_score(y_test, y_pred1)*100
#
model2 = DecisionTreeClassifier(max_depth=3)
model2.fit(X_train, y_train)
y_pred2 = model2.predict(X_test)
print(accuracy_score(y_test, y_pred2))
accuracy[str(model2)] = accuracy_score(y_test, y_pred2)*100

model3 = RandomForestClassifier(max_depth=6)
model3.fit(X_train, y_train)
y_pred3 = model3.predict(X_test)
print(accuracy_score(y_test, y_pred3))
accuracy[str(model3)] = accuracy_score(y_test, y_pred3)*100

model4 = GradientBoostingClassifier(n_estimators=300, learning_rate=0.1)
model4.fit(X_train, y_train)
y_pred4 = model4.predict(X_test)
print(accuracy_score(y_test, y_pred4))
accuracy[str(model4)] = accuracy_score(y_test, y_pred4)*100

# accuracy

algos = list(accuracy.keys())
accu_val = list(accuracy.values())

plt.bar(algos, accu_val, width=0.4)
plt.title('Accuracy Differences')
plt.xlabel('Algorithms')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.show()

# new plots of logisic regression:

param_combinations = {
'learning_rate': np.arange(0.05, 0.4, 0.05),
'max_depth': np.arange(3, 10),
'min_child_weight': np.arange(1, 7, 2),
'gamma': np.arange(0.0, 0.5, 0.1),}
XGB = xgboost.XGBClassifier()
perfect_params = RandomizedSearchCV(XGB, param_distributions=param_combinations, n_iter=6, n_jobs=-1, scoring='roc_auc')

perfect_params.fit(X, y)
perfect_params.best_params_
model5 = xgboost.XGBClassifier(min_child_weight=3, max_depth=8, learning_rate=0.05, gamma=0.0)
score = cross_val_score(model5, X, y, cv=10)
print(score)
print('Mean: ', score.mean())


