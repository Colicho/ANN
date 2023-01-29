import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

"""
n total number of samples
nHidden is the number of nodes in the hidden layer
f function used to evaluate
df f:s derivative
samples the samples as in 1a
T target values for respective sample
eta learing rate
"""
class TwoMLP:
    # constructor function
    def __init__(self, n, nHidden, f, df, samples, T, eta, momentum):
        self.n=n
        self.nHidden=nHidden
        self.f=np.vectorize(f)
        self.df=np.vectorize(df)
        self.samples = samples
        self.X = np.concatenate((samples, np.ones(shape=(1,n))), axis=0)
        self.T=T
        self.eta=eta
        self.momentum=momentum

    #computed for all samples
    def forward(self, W, V):
        #print("X ", self.X)
        hin=np.matmul(W, self.X)
        #print("hin ", hin)
        hout=np.vstack((self.f(hin), self.n*[1]))
        #print("hout ", hout)
        oin=np.matmul(V, hout)
        #print("oin ", oin)
        out=self.f(oin)
        return hin, hout, oin, out

    #computed for all samples
    def backward(self, W, V):
        values=self.forward(W, V)
        hin, hout, oin, out=values[0], values[1], values[2], values[3]
        delta_o = (out-self.T)*self.df(oin)
        delta_h=np.matmul(np.transpose(V), delta_o)
        delta_h = delta_h[:-1]
        delta_h=delta_h*self.df(hin)
        #print(np.outer(np.transpose(V), delta_o))
        #print(delta_h)
        return delta_h, delta_o, hout, out

    #batch
    def weightUpdate(self, W, V, delta_W, delta_V):
        values = self.backward(W, V)
        delta_h, delta_o, hout, out = values[0], values[1], values[2], values[3]
        delta_W = self.eta*(self.momentum*delta_W-(1-self.momentum)*np.dot(delta_h, np.transpose(self.X)))
        delta_V = self.eta*(self.momentum*delta_V-(1-self.momentum)*np.dot(delta_o, np.transpose(hout)))
        #print(delta_W, delta_V)
        #print(np.shape(delta_W), np.shape(delta_V))
        return W+delta_W,V+delta_V, delta_W, delta_V


def func_approx(nHidden, epochs):
    sigmaW = 2
    sigmaV = 2
    delta_W = 0
    delta_V = 0
    #generating variables
    x = [(-5 + i/2) for i in range(21)]
    y = [(-5 + i/2) for i in range(21)]
    z_func_vector = np.vectorize(z_func)
    x, y, = np.meshgrid(x, y)

    z = z_func_vector(x, y)
    targets = np.reshape(z, (1, len(x)*len(y)))
    patterns = np.vstack((np.reshape(x, (1, len(x)*len(y))), np.reshape(y, (1, len(x)*len(y)))))

    MLP = TwoMLP(len(x)*len(y), nHidden, f, df, patterns, targets, 0.05, 0.9)
    W = InitialWeightMatrix(nHidden, 3, sigmaW)
    V = InitialWeightMatrix(1, nHidden + 1, sigmaV)

    for i in range(epochs):
        variables = MLP.weightUpdate(W, V, delta_W, delta_V)
        W, V, delta_W, delta_V = variables[0], variables[1], variables[2], variables[3]
    variable = MLP.forward(W,V)
    print(z)
    z = np.reshape(variable[3], (len(x), len(x)))
    print(z)
    plot_func_approx(x,y,z)


def plot_func_approx(x, y, z):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()

def z_func(x, y):
    return np.exp(-x*x*0.1) * np.exp(-y*y*0.1) - 0.5

#Testing with three samples
def f(x):
    return 2/(1+np.exp(-x))-1

def df(x):
    return (1+f(x))*(1-f(x))*0.5

def InitialWeightMatrix(nrows, ncolumns, sigmaW):
    return np.reshape(np.random.standard_normal(nrows*ncolumns) * sigmaW, (nrows,ncolumns))



#W, V = weights_init(3, 2, 1)
#func_approx(2, 10)

n=100
nHidden=25
dInitial =2
epochs = 500



func_approx(nHidden, epochs)




