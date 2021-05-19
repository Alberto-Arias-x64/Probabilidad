"""
La probabilidad es una creencia que tenemos sobre la ocurrencia de eventos elementales.
La probabilidad es la cuantizacion de la incertidumbre

¿En qué casos usamos la probabilidad?

Intuitivamente, hacemos estimaciones de la probabilidad de que algo ocurra o no, al desconocimiento
que tenemos sobre la información relevante de un evento lo llamamos incertidumbre.

El azar en estos términos no existe, representa la ausencia del conocimiento de todas
las variables que componen un sistema.

En otras palabras, la probabilidad es un lenguaje que nos permite cuantificar la incertidumbre

Estadistica Frecuentista y balleciana

conjunta = considera la ocurrencia de ambos sucesos

marginal = de una compuesta sale una simple

condicional = Recla del producto, una esta breviamente definida !!no reflejan causalidad!!
"""
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
from matplotlib import colors
from numpy import hstack
from numpy.core.defchararray import array
from numpy.core.numeric import True_ 
from numpy.random import binomial
from numpy.random import normal
from scipy.stats import binom 
from scipy.stats import norm 
from mpl_toolkits.mplot3d import Axes3D
from math import factorial
from matplotlib import cm
from scipy.stats.stats import ModeResult 
from sklearn.neighbors import KernelDensity

def Distribucion_binomial():
    k=int(input("numero de exitos: "))
    n=int(input("numero de lanzamientos: "))
    p=int(input("probabilidad de exito: "))
    return factorial(n)/(factorial(k)*factorial(n-k))*pow(p,k)*pow(1-p,n-k)

def DB(rango):
    values = [0,1,2,3]
    arr = [binomial(3,0.5) for _ in range (rango)]
    sim = np.unique(arr,return_counts=True)[1]/len(arr)
    teorico = [binom(3,0.5).pmf(k) for k in values]
    plt.subplot(1,2,1)
    plt.bar(values,sim,color='r')
    plt.subplot(1,2,2)
    plt.bar(values,teorico,color='b',alpha=0.5)
    plt.show()
    return 0

def Distribucion_gaussiana():
    x = np.arange(-4,4,0.1)
    mu=0.0
    sigma=1
    y = 1/(sigma*np.sqrt(2*np.pi))*np.exp(-0.5*pow((x-mu)/sigma,2))
    plt.plot(x,y)
    plt.show()

    dist = norm(0,1)
    y = [dist.pdf(value) for value in x] # sitribucion pobabilidad de densidad
    plt.plot(x,y)
    plt.show()

    y = [dist.cdf(value) for value in x] #distribucion probabilidad acumulada
    plt.plot(x,y)
    plt.show()
    return 0

def ejersisio_alas():#campana de gauss
    df = pd.read_excel('./s057.xls')
    array1=df['Normally Distributed Housefly Wing Lengths'].values[4:]
    estimacion_dis(array1)
    valores,distribuccion=np.unique(array1,return_counts=True)
    plt.bar(valores,distribuccion/len(array1))
    plt.show()
    return 0

def estimacion_dis(array1): #estimacion parametrica
    mu = array1.mean()
    sigma = array1.std()
    x = np.arange(30,60,0.1)
    dist= norm(mu,sigma)
    y = [dist.pdf(value) for value in x]
    plt.plot(x,y,color='r')
    return 0

def distribucion_normal(): #estimacion parametrica
    sample = normal(loc=50,scale=5, size=10000)# mu = 50 ,sigma = 5
    mu = sample.mean()
    sigma = sample.std()
    dist = norm(mu, sigma)
    values = [value for value in range(30, 70)]
    probabilities = [dist.pdf(value) for value in values]
    plt.hist(sample, bins=300, density=True)
    plt.plot(values, probabilities)
    plt.show()
    return 0

def distribucion_no_parametrica():
    sample1 = normal(loc=20, scale=5 ,size=300)
    sample2 = normal(loc=40, scale=5, size=700)
    sample = hstack((sample1,sample2))

    model = KernelDensity(bandwidth=2, kernel='gaussian')
    sample = sample.reshape((len(sample),1))
    model.fit(sample)

    values = np.asarray([values for values in range(1,60)])
    values = values.reshape((len(values),1))
    probabilities = model.score_samples(values) #probabilidad logaritmica
    probabilities = np.exp(probabilities) #regresandola a normal con una exponencial

    plt.hist(sample,bins=30,density=True)
    plt.plot(values,probabilities,color='r')
    plt.show()

    return 0

def verosimilitud():
    def likelihood(y,yp):
        return yp*y+(1-yp)*(1-y)
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    Y = np.arange(0,1,0.01)
    YP = np.arange(0,1,0.01)
    Y,YP = np.meshgrid(Y,YP)
    Z = likelihood(Y,YP)

    surf = ax.plot_surface(Y, YP, Z, cmap=cm.coolwarm)
    fig.colorbar(surf,shrink=0.5,aspect=5)
    plt.show()

def vero_Log(): #maximo de los logaritmos
    pass

verosimilitud()
