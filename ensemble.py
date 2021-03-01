import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, validation_curve

df = pd.read_csv("data_2.csv", header = None)
X = df.iloc[:, 0:6].values
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
Y = df.iloc[:, -1].values

tree_depth = [1, 5, 25, 100, 1000]
number_of_features = [1, 2, 3, 4, 5, 6]
learning_rate = [0.1, 0.05, 0.01, 0.005, 0.001]
number_of_estimators = np.arange(1, 101, 10)

test_scores_mean_all = np.ndarray((len(number_of_estimators), len(number_of_features)))

train_scores_mean_all = np.ndarray((len(number_of_estimators), len(number_of_features)))

for td in tree_depth:
    for nf in number_of_features:
        model = RandomForestClassifier(max_depth=td, max_features=nf)
        train_scores, test_scores = validation_curve(
                model, X, Y,
                param_name='n_estimators',
                param_range=number_of_estimators,
                scoring='f1')
        train_scores_mean_all[:, nf-1] = np.mean(train_scores, axis=1)
        test_scores_mean_all[:, nf-1] = np.mean(test_scores, axis=1)

    plt.title("Trening kriva, velicina stabla = "+str(td))
    plt.xlabel("Velicina ansambla")
    plt.ylabel("Skor")
    lw = 6
    plt.plot(number_of_estimators, train_scores_mean_all[:, 0], label="Maksimalni broj odlika: 1")
    plt.plot(number_of_estimators, train_scores_mean_all[:, 1], label="Maksimalni broj odlika: 2")
    plt.plot(number_of_estimators, train_scores_mean_all[:, 2], label="Maksimalni broj odlika: 3")
    plt.plot(number_of_estimators, train_scores_mean_all[:, 3], label="Maksimalni broj odlika: 4")
    plt.plot(number_of_estimators, train_scores_mean_all[:, 4], label="Maksimalni broj odlika: 5")
    plt.plot(number_of_estimators, train_scores_mean_all[:, 5], label="Maksimalni broj odlika: 6")
    plt.legend()
    plt.savefig('RF_trening_kriva_'+str(td)+'.png')
    plt.close()

    plt.title("Test kriva, velicina stabla = "+str(td))
    plt.xlabel("Velicina ansambla")
    plt.ylabel("Skor")
    lw = 6
    plt.plot(number_of_estimators, test_scores_mean_all[:, 0], label="Maksimalni broj odlika: 1")
    plt.plot(number_of_estimators, test_scores_mean_all[:, 1], label="Maksimalni broj odlika: 2")
    plt.plot(number_of_estimators, test_scores_mean_all[:, 2], label="Maksimalni broj odlika: 3")
    plt.plot(number_of_estimators, test_scores_mean_all[:, 3], label="Maksimalni broj odlika: 4")
    plt.plot(number_of_estimators, test_scores_mean_all[:, 4], label="Maksimalni broj odlika: 5")
    plt.plot(number_of_estimators, test_scores_mean_all[:, 5], label="Maksimalni broj odlika: 6")
    plt.legend()
    plt.savefig('RF_test_kriva_'+str(td)+'.png')
    plt.close()

test_scores_mean_all = np.zeros((len(number_of_estimators), len(learning_rate)))

train_scores_mean_all = np.zeros((len(number_of_estimators), len(learning_rate)))
for td in tree_depth:
    i = 0
    for lr in learning_rate:
        model = GradientBoostingClassifier(max_depth=td, learning_rate=lr)
        train_scores, test_scores = validation_curve(
                model, X, Y,
                param_name='n_estimators',
                param_range=number_of_estimators,
                scoring='f1')
        train_scores_mean_all[:, i] = np.mean(train_scores, axis=1)
        test_scores_mean_all[:, i] = np.mean(test_scores, axis=1)
        i+=1

    plt.title("Trening kriva, velicina stabla = "+str(td))
    plt.xlabel("Velicina ansambla")
    plt.ylabel("Skor")
    lw = 6
    plt.plot(number_of_estimators, train_scores_mean_all[:, 0], label="Stopa ucenja: 0.1")
    plt.plot(number_of_estimators, train_scores_mean_all[:, 1], label="Stopa ucenja: 0.05")
    plt.plot(number_of_estimators, train_scores_mean_all[:, 2], label="Stopa ucenja: 0.01")
    plt.plot(number_of_estimators, train_scores_mean_all[:, 3], label="Stopa ucenja: 0.005")
    plt.plot(number_of_estimators, train_scores_mean_all[:, 4], label="Stopa ucenja: 0.001")
    plt.legend()
    plt.savefig('GB_trening_kriva_'+str(td)+'.png')
    plt.close()

    plt.title("Test kriva, velicina stabla = "+str(td))
    plt.xlabel("Velicina ansambla")
    plt.ylabel("Skor")
    lw = 6
    plt.plot(number_of_estimators, test_scores_mean_all[:, 0], label="Stopa ucenja: 0.1")
    plt.plot(number_of_estimators, test_scores_mean_all[:, 1], label="Stopa ucenja: 0.05")
    plt.plot(number_of_estimators, test_scores_mean_all[:, 2], label="Stopa ucenja: 0.01")
    plt.plot(number_of_estimators, test_scores_mean_all[:, 3], label="Stopa ucenja: 0.005")
    plt.plot(number_of_estimators, test_scores_mean_all[:, 4], label="Stopa ucenja: 0.001")
    plt.legend()
    plt.savefig('GB_test_kriva_'+str(td)+'.png')
    plt.close()

print("FINISHED")