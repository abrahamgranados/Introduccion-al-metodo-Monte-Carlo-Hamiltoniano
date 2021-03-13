import numpy as np
import random
import math
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats 
from scipy.stats import multivariate_normal
from google.colab import files
from statsmodels.graphics.tsaplots import plot_acf
import time

y=np.array([[2.8,0.8,-0.3,0.7,-0.1,0.1,1.8,1.2]])
desviacion_ejemplo=np.array([[0.8,0.5,0.8,0.6,0.5,0.6,0.5,0.4]])

def U(parametros):   #Distribucion log  posterior
  eta, mu2, tau = parametros[0][0:8],parametros[0][8], parametros[0][9] 
  log_prior = -np.sum(scipy.stats.norm(0, 1).logpdf(parametros))    
  log_likelihood = -np.sum(scipy.stats.norm(mu2 + eta *tau, desviacion_ejemplo).logpdf(y))  
  return(log_prior + log_likelihood)

def grad_U(parametros):   #Gradiente de la distribución objetivo
    eta, mu2, tau = parametros[0][0:8],parametros[0][8], parametros[0][9] 
    d_tau = tau - np.sum(((y-(mu2+tau*eta))/desviacion_ejemplo**2)*eta)  #los parametros de los que se va a hacer inferencia
    d_mu = mu2 - np.sum(((y-(mu2+tau*eta))/desviacion_ejemplo**2))
    d_eta= eta - ((y-(mu2+tau*eta))/desviacion_ejemplo**2)*(tau)
    return(np.concatenate((d_eta, np.array([[d_mu]]),np.array([[d_tau]]) ), axis=1))

def HMCMC(U, grad_U, epsilon, L, inicial_q, N):
  #En el primer paso nuevos valores son  escogidos para el momentum, p, aleatoriamente de una distribución normal, y son independientes de la posición q. 
  mu, sigma = 0, 1 # Parametros de la normal
  
  q = inicial_q #El punto de inicio del algoritmo
  cadena=q
  rechazos=0

  for i in range(1,N): 

    p = np.array([np.random.normal(mu, sigma, q.shape[1])])  #Asignamos el nuevo valor de p, independiente de q

    actual_p = p  
    p=p- epsilon * grad_U(q) / 2
    for i in range(1,L+1):
      q= q+epsilon*p
      if(i!=L):
        p=p-epsilon*grad_U(q)
    p= p - epsilon * grad_U(q) / 2   ##termina el algoritmo de Leap Frog 
  # Hacemos la propuesta simétrica cambiando el signo 
    p = -p
    actual_U = U(inicial_q)
    actual_K = np.sum(actual_p**2) / 2
    propuesta_U = U(q)
    propuesta_K = np.sum(p**2) / 2
    if random.uniform(0, 1) < np.exp(actual_U-propuesta_U+actual_K-propuesta_K):
      cadena=np.r_[cadena,q]
      inicial_q=q
      print(len(cadena))

    else:
      cadena=np.r_[cadena,inicial_q]
      rechazos=rechazos+1
     
  return(cadena, rechazos)

np.random.seed(0)
random.seed(0)
t0 = time.time()
AA=HMCMC(U,grad_U,0.08,60,np.array([[2,2,2,2,2,2,2,2,2,2]]),500000)
t1 = time.time()
print("--- %s segundos ---" % (t1- t0))

1-AA[1]/len(AA[0]-1)   #probabilidad de aceptacion

vector_verosim2 = np.zeros(300)
for i in range(0,300):
  if i==0:
    logverosimilitud=-U(np.array([AA[0][i]]))
  else:
    logverosimilitud=np.r_[logverosimilitud,-U(np.array([AA[0][i]]))]

burnin=100
plt.grid(linestyle='dashed')
plt.plot(logverosimilitud)
plt.ylabel('$log(S(X_{t}))$')
plt.xlabel('$X_{t}$')
plt.axvline(burnin,-100,100,  color="red",  linestyle='dashed')
plt.savefig("burnhmc1.png",bbox_inches='tight',dpi=300)
files.download("burnhmc1.png")

tau_sb = AA[0][:,9][burnin:len(AA[0]) ]#[np.arange(0,450000,50)] 
mu_sb=AA[0][:,8][burnin:len(AA[0]) ]#[np.arange(0,450000,50)] 
eta1_sb=AA[0][:,0][burnin:len(AA[0]) ]#[np.arange(0,450000,50)] <

len(eta1_sb)

fig=plot_acf(mu_sb, lags=100,alpha=0.05)  #autocorrelaciones 
plt.grid(linestyle='dashed')
plt.title("")
plt.ylabel('Autocorrelación')
plt.xlabel('Lag')

#fig.savefig("autocor_1.png",bbox_inches='tight',dpi=300)
#files.download("autocor_1.png")

fig=plot_acf(tau_sb lags=100,alpha=0.05)  #autocorrelaciones 
plt.grid(linestyle='dashed')
plt.title("")
plt.ylabel('Autocorrelación')
plt.xlabel('Lag')

#fig.savefig("tauhmc.png",bbox_inches='tight',dpi=300)
#files.download("tauhmc.png")

fig=plot_acf(eta1_sb, lags=100,alpha=0.05)  #autocorrelaciones 
plt.grid(linestyle='dashed')
plt.title("")
plt.ylabel('Autocorrelación')
plt.xlabel('Lag')

lag=35
tau_sl = tau_sb[np.arange(0,len(tau_sb),lag )] 
mu_sl=mu_sb[np.arange(0,len(mu_sb),lag )]
eta1_sl=eta1_sb[np.arange(0,len(eta1_sb),lag )]

len(tau_sl)

sns.kdeplot(mu_sl, tau_sl, fill=True, levels=20)
plt.grid(linestyle='dashed')
plt.ylabel(r'$\tau$')
plt.xlabel('$\mu$')
plt.savefig("unohmc.png",bbox_inches='tight',dpi=300)
files.download("unohmc.png")

sns.kdeplot( tau_sl,eta1_sl, fill=True, levels=20)
plt.grid(linestyle='dashed')
plt.xlabel(r'$\tau$')
plt.ylabel(r'$\eta_{1}$')
plt.savefig("doshmc.png",bbox_inches='tight',dpi=300)
files.download("doshmc.png")

"""##Simulacion de las 500 muestras de la distribución objetivo

"""

np.random.seed(0)
random.seed(0)
t0 = time.time()
AA=HMCMC(U,grad_U,0.08,60,np.array([[2,2,2,2,2,2,2,2,2,2]]),100+35*500)
t1 = time.time()
print("--- %s segundos ---" % (t1- t0))

1-AA[1]/len(AA[0]-1)

burnin=100
lag=35
tau_sl = AA[0][:,9][burnin:len(AA[0]) ][np.arange(0, len(AA[0]) -burnin,lag )] 
mu_sl=AA[0][:,8][burnin:len(AA[0]) ][np.arange(0, len(AA[0]) -burnin,lag )]
eta1_sl=AA[0][:,0][burnin:len(AA[0]) ][np.arange(0, len(AA[0]) -burnin,lag )]

len(eta1_sl)

sns.kdeplot(mu_sl, tau_sl, fill=True, levels=20)
plt.grid(linestyle='dashed')
plt.ylabel(r'$\tau$')
plt.xlabel('$\mu$')
plt.xlim(-1, 2)  
plt.ylim(-2.2, 2.3)  

#plt.savefig("unohmc500.png",bbox_inches='tight',dpi=300)
#files.download("unohmc500.png")

sns.kdeplot( tau_sl,eta1_sl, fill=True, levels=20)
plt.grid(linestyle='dashed')
plt.xlabel(r'$\tau$')
plt.ylabel(r'$\eta_{1}$')
plt.xlim(-2.8, 2.5)  
plt.ylim(-4.2, 5)  
plt.savefig("doshmc500.png",bbox_inches='tight',dpi=300)
files.download("doshmc500.png")

"""#RWMH"""

#RANDOM WALK METROPOLIS HASTINGS                          
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss
from google.colab import files
import scipy.stats 
from scipy.stats import kde
from statsmodels.graphics.tsaplots import plot_acf
import time

y=np.array([[2.8,0.8,-0.3,0.7,-0.1,0.1,1.8,1.2]])
sigma2=np.array([[0.8,0.5,0.8,0.6,0.5,0.6,0.5,0.4]])

def U(parametros):   #Distribucion log  posterior
  eta, mu2, tau = parametros[0][0:8],parametros[0][8], parametros[0][9]   #los parametros de los que se va a hacer inferencia
  log_prior = np.sum(scipy.stats.norm(0, 1).logpdf(parametros))    #posterior P(thetha \mid mu2, tau)
  log_likelihood = np.sum(scipy.stats.norm(mu2 + eta *tau, sigma2).logpdf(y))  #posterior P(y \mid tau, sigma2) la de interés
  return(log_prior + log_likelihood) 

def razonf(q1,q2):
  return(np.exp(U(q1)-U(q2)))


def U2(parametros):   #Distribucion log  posterior
  eta, mu2, tau = parametros[0][0:8],parametros[0][8], parametros[0][9]   #los parametros de los que se va a hacer inferencia
  log_prior = np.prod(scipy.stats.norm(0, 1).pdf(parametros))    #posterior P(thetha \mid mu2, tau)
  log_likelihood = np.prod(scipy.stats.norm(mu2 + eta * tau, sigma2).pdf(y))  #posterior P(y \mid tau, sigma2) la de interés
  return(log_prior * log_likelihood) 


def razonf2(q1,q2):
  return(U2(q1)/U2(q2))



def RWMH_dim(p_ini, mu, sigma, N):
  rechazos=0     #vector de muestras
  muestra=p_ini        #primera posición nuestro punto inicial
  for i in range(1, N):           
    propuesta=p_ini+np.random.normal(mu, sigma, p_ini.shape[1])    #el valor propuesto 
    razon = min(razonf(propuesta, p_ini),1)
    if random.uniform(0, 1) < razon:
      actual=propuesta
      p_ini=propuesta
    else:
      actual=p_ini
      rechazos=rechazos+1
      print(i)
    muestra=np.r_[muestra,actual]
  #uno es el punto inicial y en la ultima posicion esta el porcentaje
  return(muestra, rechazos)

inicial=np.array([[2,2,2,2,2,2,2,2,2,2]])

np.random.seed(0)
random.seed(0)
t0 = time.time()
muestra_RWMH = RWMH_dim(inicial,0,0.32,500000) 
t1 = time.time()
print("--- %s segundos ---" % (t1- t0))

1-muestra_RWMH[1]/(len(muestra_RWMH[0]))

vector_verosim2 = np.zeros(300)
for i in range(0,300):
  if i==0:
    logverosimilitud=U(np.array([muestra_RWMH[0][i]]))
  else:
    logverosimilitud=np.r_[logverosimilitud, U(np.array([muestra_RWMH[0][i]]))   ]

burnin=200
plt.grid(linestyle='dashed')
plt.plot(logverosimilitud)
plt.ylabel('$log(S(X_{t})$')
plt.xlabel('$X_{t}$')
plt.axvline(burnin,-100,100,  color="red",  linestyle='dashed')
#plt.savefig("burnrw8.png",bbox_inches='tight',dpi=300)
#files.download("burnrw8.png")

burnin=200
tau_sb = muestra_RWMH[0][:,9][burnin:len(muestra_RWMH[0])] 
mu_sb=muestra_RWMH[0][:,8][burnin:len(muestra_RWMH[0])] 
eta1_sb=muestra_RWMH[0][:,0][burnin:len(muestra_RWMH[0])]

len(tau_sb)

fig=plot_acf(mu_sb, lags=1000,alpha=0.05)  #autocorrelaciones 
plt.grid(linestyle='dashed')
plt.title("")
plt.ylabel('Autocorrelación')
plt.xlabel('Lag')

#fig.savefig("autocor_1.png",bbox_inches='tight',dpi=300)
#files.download("autocor_1.png")

fig=plot_acf(tau_sb, lags=4000,alpha=0.05)  #autocorrelaciones 
plt.grid(linestyle='dashed')
plt.title("")
plt.ylabel('Autocorrelación')
plt.xlabel('Lag')

#fig.savefig("tau_rw.png",bbox_inches='tight',dpi=300)
#files.download("tau_rw.png")

fig=plot_acf(eta1_sb, lags=4000,alpha=0.05)  #autocorrelaciones 
plt.grid(linestyle='dashed')
plt.title("")
plt.ylabel('Autocorrelación')
plt.xlabel('Lag')
#fig.savefig("autocor_1.png",bbox_inches='tight',dpi=300)
#files.download("autocor_1.png")

lag=2300
tau_sl = tau_sb[np.arange(0,len(tau_sb),lag )] 
mu_sl=mu_sb[np.arange(0,len(mu_sb),lag )]
eta1_sl=eta1_sb[np.arange(0,len(eta1_sb),lag )]

len(tau_sl)

sns.kdeplot(mu_sl, tau_sl, fill=True, levels=20)
plt.grid(linestyle='dashed')
plt.ylabel(r'$\tau$')
plt.xlabel('$\mu$')
plt.xlim(-1, 2)  
plt.ylim(-2.2, 2.3)  
plt.savefig("unorw8.png",bbox_inches='tight',dpi=300)
files.download("unorw8.png")

sns.kdeplot( tau_sl,eta1_sl, fill=True, levels=20)
plt.grid(linestyle='dashed')

plt.xlabel(r'$\tau$')
plt.ylabel(r'$\eta_{1}$')
plt.xlim(-2.8, 2.5)  
plt.ylim(-4.2, 5)  
#plt.savefig("dosrw8.png",bbox_inches='tight',dpi=300)
#files.download("dosrw8.png")

"""##Simulacion de las 500 muestras de la distribución objetivo

"""

np.random.seed(0)
random.seed(0)
t0 = time.time()
muestra_RWMH = RWMH_dim(inicial,0,0.32,200+2300*500) 
t1 = time.time()
print("--- %s segundos ---" % (t1- t0))

1-muestra_RWMH[1]/(len(muestra_RWMH[0]))

burnin, lag=200, 2300
tau_sl = muestra_RWMH[0][:,9][burnin:len(muestra_RWMH[0])][np.arange(0,len(muestra_RWMH[0])-burnin,lag )]
mu_sl=muestra_RWMH[0][:,8][burnin:len(muestra_RWMH[0])][np.arange(0,len(muestra_RWMH[0])-burnin,lag )]
eta1_sl=muestra_RWMH[0][:,0][burnin:len(muestra_RWMH[0])][np.arange(0,len(muestra_RWMH[0])-burnin,lag )]

sns.kdeplot(mu_sl, tau_sl, fill=True, levels=20)
plt.grid(linestyle='dashed')
plt.ylabel(r'$\tau$')
plt.xlabel('$\mu$')
plt.xlim(-1, 2)  
plt.ylim(-2.2, 2.3)  
#plt.savefig("unorw8500.png",bbox_inches='tight',dpi=300)
#files.download("unorw8500.png")

sns.kdeplot( tau_sl,eta1_sl, fill=True, levels=20)
plt.grid(linestyle='dashed')

plt.xlabel(r'$\tau$')
plt.ylabel(r'$\eta_{1}$')
plt.xlim(-2.8, 2.5)  
plt.ylim(-4.2, 5)  
plt.savefig("dosrw8500.png",bbox_inches='tight',dpi=300)
files.download("dosrw8500.png")

"""# t-walk"""

!pip install '/content/pytwalk-1.6.0 (4).tar.gz'

import pytwalk as pt

y=np.array([[2.8,0.8,-0.3,0.7,-0.1,0.1,1.8,1.2]])
desviacion_ejemplo=np.array([[0.8,0.5,0.8,0.6,0.5,0.6,0.5,0.4]])

def verosimilitud(parametros):   #Distribucion log  posterior
  eta, mu2, tau = parametros[0:8],parametros[8], parametros[9] 
  log_prior = np.sum(scipy.stats.norm(0, 1).logpdf(parametros))    
  log_likelihood = np.sum(scipy.stats.norm(mu2 + eta *tau, desviacion_ejemplo).logpdf(y))  
  return(log_prior + log_likelihood)  


def fU(parametros):   #Distribucion log  posterior
  eta, mu2, tau = parametros[0:8],parametros[8], parametros[9] 
  log_prior = -np.sum(scipy.stats.norm(0, 1).logpdf(parametros))    
  log_likelihood = -np.sum(scipy.stats.norm(mu2 + eta *tau, desviacion_ejemplo).logpdf(y))  
  return(log_prior + log_likelihood)

def fUSupp(x):
	return all( x)

###          we define the objective function with the U function
###          which is -log of the density function.
###          The support is defined in a separate function.
###   The dimension of the parameter space is n
hh= pt.pytwalk( n=5, U=fU, Supp=fUSupp)

A=pt.pytwalk( n=10, U=fU, Supp=fUSupp)

np.random.seed(0)
random.seed(0)
t0 = time.time()
A.Run( T=500000, x0=3*np.ones(10), xp0=2*np.ones(10))
t1 = time.time()
print("--- %s segundos ---" % (t1- t0))

A.Ana()

AA=A.Output

vector_verosim2 = np.zeros(1000)
for i in range(0,1000):

  if i==0:
    logverosimilitud=verosimilitud(AA[i][0:10])
  else:
    logverosimilitud=np.r_[logverosimilitud,verosimilitud(AA[i][0:10])]

burnin=200
plt.grid(linestyle='dashed')
plt.plot(logverosimilitud)
plt.xlabel('$X_{t}$')
plt.axvline(burnin,-100,100,  color="red",  linestyle='dashed')
#plt.savefig("burntw.png",bbox_inches='tight',dpi=300)
#files.download("burntw.png")

tau_sb = AA[:,9][burnin:len(AA[:,0])]
mu_sb=AA[:,8][burnin:len(AA[:,0])] 
eta1_sb=AA[:,0][burnin:len(AA[:,0])]

len(tau_sb)

fig=plot_acf(mu_sb, lags=10000,alpha=0.05)  #autocorrelaciones 
plt.grid(linestyle='dashed')
plt.title("")
plt.ylabel('Autocorrelación')
plt.xlabel('Lag')

#fig.savefig("autocor_1.png",bbox_inches='tight',dpi=300)
#files.download("autocor_1.png")

fig=plot_acf(tau_sb, lags=11000, alpha=0.05)  #autocorrelaciones 
plt.grid(linestyle='dashed')
plt.title("")

plt.xlabel('Lag')
fig.savefig("tautw.png",bbox_inches='tight',dpi=300)
files.download("tautw.png")

fig=plot_acf(eta1_sb, lags=10000,alpha=0.05)  #autocorrelaciones 
plt.grid(linestyle='dashed')
plt.title("")
plt.ylabel('Autocorrelación')
plt.xlabel('Lag')

lag=9000
tau_sl = tau_sb[np.arange(0,len(tau_sb),lag )] 
mu_sl=mu_sb[np.arange(0,len(mu_sb),lag )]
eta1_sl=eta1_sb[np.arange(0,len(eta1_sb),lag )]

sns.kdeplot(mu_sl, tau_sl, fill=True, levels=20)
plt.grid(linestyle='dashed')
plt.ylabel(r'$\tau$')
plt.xlabel('$\mu$')
plt.xlim(-1, 2)  
plt.ylim(-2.4, 2.3)  
plt.savefig("unotw.png",bbox_inches='tight',dpi=300)
files.download("unotw.png")

sns.kdeplot( tau_sl,eta1_sl, fill=True, levels=20)
plt.grid(linestyle='dashed')
plt.xlabel(r'$\tau$')
plt.ylabel(r'$\eta_{1}$')
plt.xlim(-2.8, 2.5)  
plt.ylim(-4.2, 5)  
plt.savefig("dostw.png",bbox_inches='tight',dpi=300)
files.download("dostw.png")

len(tau_sl)

"""##Simulacion de las 500 muestras de la distribución objetivo"""

np.random.seed(0)
random.seed(0)
t0 = time.time()
A.Run( T=500*9000+200, x0=3*np.ones(10), xp0=2*np.ones(10))
t1 = time.time()
print("--- %s segundos ---" % (t1- t0))

AA=A.Output

A.Ana()

burnin, lag=200, 9000
tau_sl = AA[:,9][burnin:len(AA[:,0])][np.arange(0,len(AA[:,0])-burnin,lag )] 
mu_sl=AA[:,8][burnin:len(AA[:,0])][np.arange(0,len(AA[:,0])-burnin,lag )] 
eta1_sl=AA[:,0][burnin:len(AA[:,0])][np.arange(0,len(AA[:,0])-burnin,lag )]

sns.kdeplot(mu_sl, tau_sl, fill=True, levels=20)
plt.grid(linestyle='dashed')
plt.ylabel(r'$\tau$')
plt.xlabel('$\mu$')
plt.xlim(-1, 2)  
plt.ylim(-2.2, 2.3)  
#plt.savefig("unotw500.png",bbox_inches='tight',dpi=300)
#files.download("unotw500.png")

sns.kdeplot( tau_sl,eta1_sl, fill=True, levels=20)
plt.grid(linestyle='dashed')
plt.xlabel(r'$\tau$')
plt.ylabel(r'$\eta_{1}$')
plt.xlim(-2.8, 2.5)  
plt.ylim(-4.2, 5)  
#plt.savefig("dostw500.png",bbox_inches='tight',dpi=300)
#files.download("dostw500.png")
