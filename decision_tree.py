import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, validation_curve

df = pd.read_csv("data_1.csv", header = None)
X = df.iloc[:, [11, 12]].values
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
Y = df.iloc[:, -1].values

tree_depth = np.arange(1, 50, 2)
train_scores, test_scores = validation_curve(
                DecisionTreeClassifier(), X, Y,
                param_name='max_depth',
                param_range=tree_depth,
                scoring='f1')

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title("Validaciona kriva")
plt.xlabel("Dubina stabla")
plt.ylabel("Skor")
lw = 2
plt.plot(tree_depth, train_scores_mean, label="Trening",
            color="darkorange")
plt.fill_between(tree_depth, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange")
plt.plot(tree_depth, test_scores_mean, label="Validacija",
             color="navy")
plt.fill_between(tree_depth, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy")
plt.legend(loc="best")
plt.show()

def plot_decision_boundary(model, xmin, xmax, ymin, ymax):
  xx, yy = np.meshgrid(
      np.linspace(xmin, xmax, num=100, endpoint=True), 
      np.linspace(ymin, ymax, num=100, endpoint=True))

  Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
  Z = Z.reshape(xx.shape)
  plt.contourf(xx, yy, Z, alpha=0.2, cmap='bwr')

model1 = DecisionTreeClassifier(max_depth = 4)
model1.fit(X, Y)

plt.scatter(X[:, 0], X[:, 1], c=Y)
plt.axis('equal')

xmin, xmax, ymin, ymax = plt.axis()
plot_decision_boundary(model1, xmin, xmax, ymin, ymax)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Granica odluke - pravilno obucen')
plt.show()

model2 = DecisionTreeClassifier(max_depth = 1000)
model2.fit(X, Y)

plt.scatter(X[:, 0], X[:, 1], c=Y)
plt.axis('equal')

xmin, xmax, ymin, ymax = plt.axis()
plot_decision_boundary(model2, xmin, xmax, ymin, ymax)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Granica odluke - preobucen')
plt.show()

model3 = DecisionTreeClassifier(max_depth = 1)
model3.fit(X, Y)

plt.scatter(X[:, 0], X[:, 1], c=Y)
plt.axis('equal')

xmin, xmax, ymin, ymax = plt.axis()
plot_decision_boundary(model3, xmin, xmax, ymin, ymax)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Granica odluke - podobucen')
plt.show()