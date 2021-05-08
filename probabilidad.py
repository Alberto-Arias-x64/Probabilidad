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
from numpy.random import binomial
from scipy.stats import binom 
from math import factorial 
import matplotlib.pyplot as plt 

def Distribucion_binomial(k,n,p):
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
    return


#k= numero de exitos
#n= numero de lansamientos
#p= probabilidad de exito
"""
k=int(input("Numero 1"))
n=int(input("Numero 2"))
p=int(input("Numero 3"))
print(Distribucion_binomial(k,n,p))
"""

arr=DB(100000)