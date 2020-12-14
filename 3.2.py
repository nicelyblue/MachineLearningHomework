from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
from matplotlib import pyplot
import numpy, csv, pandas

def kernel(x):
    return numpy.dot(x, x.T)

def fit(X, y, C = 10):
    m = X.shape[0]
    y = y.reshape(-1,1) * 1.
    X_y = y * X
    H = kernel(X_y) * 1.

    P = cvxopt_matrix(H)
    q = cvxopt_matrix(-numpy.ones((m, 1)))
    G = cvxopt_matrix(numpy.vstack((numpy.eye(m)*-1,numpy.eye(m))))
    h = cvxopt_matrix(numpy.hstack((numpy.zeros(m), numpy.ones(m) * C)))
    A = cvxopt_matrix(y.reshape(1, -1))
    b = cvxopt_matrix(numpy.zeros(1))

    sol = cvxopt_solvers.qp(P, q, G, h, A, b)
    alphas = numpy.array(sol['x'])

    w = ((y * alphas).T @ X).reshape(-1,1)
    S = (alphas > 1e-4).flatten()
    b = y[S] - numpy.dot(X[S], w)
    b = b[0]

    return w, b, S

def predict(x, w, b):
    h = w.T @ x + b
    return numpy.sign(h)


X = numpy.ndarray(shape = (50, 2))
y = numpy.ndarray(shape = (50, 1))

with open('svmData_ls.txt', newline = '') as data:
    reader = csv.reader(data, delimiter='\t')
    i = 0
    for row in reader:
        X[i, :] = row[0:2]
        y[i, :] = row[2]
        i+=1


w, b, S = fit(X, y, 10)
support_vectors = X[S]

sep_line = lambda x, b, w : -(b + w[0, 0] * x) / w[1, 0]

pyplot.scatter(X[:, 0], X[:, 1], c=y)
pyplot.axis('equal')

xl, xr = pyplot.xlim()
x = numpy.array([xl, xr])
y_svm = sep_line(x, b, w)
pyplot.plot(x, y_svm, '--', label='SVM')
pyplot.scatter(support_vectors[:, 0], support_vectors[:, 1], 
        s=100, marker='p', facecolors='none', edgecolor='green', linewidth=2)
pyplot.legend()
pyplot.xlabel('$x_1$')
pyplot.ylabel('$x_2$')
pyplot.title('Separaciona prava i noseci vektori')
pyplot.show()
