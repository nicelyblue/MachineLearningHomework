from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
from sklearn.metrics.pairwise import rbf_kernel
from matplotlib import pyplot
import numpy, csv, pandas

def kernel(x, y):
    gamma = 3.5
    return rbf_kernel(x, y, gamma)

def fit(X, y, C = 15):
    m = X.shape[0]
    y = y.reshape(-1,1) * 1.
    H = y * kernel(X, X) * y.T * 1.

    P = cvxopt_matrix(H)
    q = cvxopt_matrix(-numpy.ones((m, 1)))
    G = cvxopt_matrix(numpy.vstack((numpy.eye(m)*-1,numpy.eye(m))))
    h = cvxopt_matrix(numpy.hstack((numpy.zeros(m), numpy.ones(m) * C)))
    A = cvxopt_matrix(y.reshape(1, -1))
    b = cvxopt_matrix(numpy.zeros(1))

    sol = cvxopt_solvers.qp(P, q, G, h, A, b)
    alphas = numpy.array(sol['x'])
    S = (alphas > 1e-4).flatten()
    b = y[S] - ((y[S] * alphas[S]).T @ kernel(X[S], X[S])).reshape(-1, 1)
    b = b[0]
    return alphas, b, S

def predict(X, y, alphas, b, S, x):
    h = ((y[S] * alphas[S]).T @ kernel(X[S], x)).reshape(-1, 1) + b
    return numpy.sign(h)

X = numpy.ndarray(shape = (100, 2))
y = numpy.ndarray(shape = (100, 1))

with open('svmData_nls.txt', newline = '') as data:
    reader = csv.reader(data, delimiter='\t')
    i = 0
    for row in reader:
        X[i, :] = row[0:2]
        y[i, :] = row[2]
        i+=1


alphas, b, S = fit(X, y)
support_vectors = X[S]

pyplot.scatter(X[:, 0], X[:, 1], c=y)

# pomoćna f-ja za prikaz regiona odluke
def plot_decision_boundary(X, y, alphas, b, S, xmin, xmax, ymin, ymax):
  xx, yy = numpy.meshgrid(
      numpy.linspace(xmin, xmax, num=100, endpoint=True), 
      numpy.linspace(ymin, ymax, num=100, endpoint=True))
  Z = predict(X, y, alphas, b, S, numpy.c_[xx.ravel(), yy.ravel()])
  Z = Z.reshape(xx.shape)
  cs = pyplot.contourf(xx, yy, Z, alpha=0.2, cmap='bwr')
  
# grafički prikaz
xmin, xmax, ymin, ymax = pyplot.axis()
plot_decision_boundary(X, y, alphas, b, S, xmin, xmax, ymin, ymax)
pyplot.scatter(support_vectors[:, 0], support_vectors[:, 1], 
        s=100, marker='p', facecolors='none', edgecolor='green', linewidth=2)
pyplot.xlabel('$x_1$')
pyplot.ylabel('$x_2$')
pyplot.title('Separaciona prava i noseći vektori')
pyplot.show()
