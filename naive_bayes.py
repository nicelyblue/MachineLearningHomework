import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def fit(x, y, classes):
    n_samples, n_features = x.shape
    n_classes = len(classes)

    means = np.zeros((n_classes, n_features), dtype=np.float64)
    variances = np.zeros((n_classes, n_features), dtype=np.float64)
    priors = np.zeros(n_classes, dtype=np.float64)

    # racunamo srednje vrednosti i varijanse feature-a za svaku klasu i apriorne verovatnoce svake klase
    for c in classes:
        x_c = x[y == c]
        means[c, :] = x_c.mean(axis=0)
        variances[c, :] = x_c.var(axis=0)
        priors[c] = x_c.shape[0] / float(n_samples)

    return means, variances, priors

def predict(X, classes, means, variances, priors):
    y_pred = [predict_single(x, classes, means, variances, priors) for x in X]
    return np.array(y_pred)

def predict_single(x, classes, means, variances, priors):
    posteriors = []

    # racunamo verovatnoce da opservacija pripada svakoj od klasa, koristimo logaritam 'chain-rule' izraza
    for c in classes:
        prior = np.log(priors[c])
        posterior = np.sum(np.log(pdf(c, x, means, variances)))
        posterior = prior + posterior
        posteriors.append(posterior)

    return classes[np.argmax(posteriors)]

def pdf(class_, x, means, variances):
    # pomocna funkcija koja generise Gausovu raspodelu za ulaznu promenljivu 'x', nju koristimo kao
    mean = means[class_]
    var = variances[class_]
    numerator = np.exp(- (x - mean) ** 2 / (2 * var))
    denominator = np.sqrt(2 * np.pi * var)
    return numerator / denominator

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    accuracy = 100*accuracy
    print('The model is ' + str(accuracy) + '% accurate')

df = pd.read_csv("multiclass_data.csv", header=None)

X = df.iloc[:, 0:5].values

Y = df.iloc[:, -1].values

classes = np.unique(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.95, shuffle=False)

means, variances, priors = fit(X_train, Y_train, classes)

y_prediction = predict(X_test, classes, means, variances, priors)

accuracy(Y_test, y_prediction)
