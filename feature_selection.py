import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("data_1.csv", header = None)
X = df.iloc[:, 0:13].values
Y = df.iloc[:, -1].values

cor = df.corr()
sns.heatmap(cor, annot=True)
plt.show()

cor_target = abs(cor[13])
relevant_features = cor_target[cor_target>0.5]
print(relevant_features)

print(df[[0, 5]].corr())
print(df[[0, 6]].corr())
print(df[[0, 9]].corr())
print(df[[0, 12]].corr())
print(df[[0, 9]].corr())
print(df[[5, 6]].corr())
print(df[[5, 9]].corr())
print(df[[5, 12]].corr())
print(df[[6, 9]].corr())
print(df[[6, 12]].corr())
print(df[[9, 12]].corr())

log = LogisticRegression(penalty = 'none')
sfs = SequentialFeatureSelector(log, n_features_to_select = 2)
sfs.fit(X, Y)

print(sfs.get_support())
