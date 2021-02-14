import matplotlib.pyplot as plt
import numpy as np
from google.colab import files
from mpl_toolkits import mplot3d
# %matplotlib inline
A=10
m=2
k=30
w=np.sqrt(k/m)
t = np.arange(0,2*(np.pi)/w, 0.01)
cinetica=(1/2)*m*(A*w*np.cos(w*t))**2
potencial=1/2*k*(A*np.sin(w*t))**2
plt.plot(t,cinetica, 'teal',t, potencial, 'maroon',t,cinetica+potencial, 'darkslategray')
plt.ylabel('Energía')
plt.xlabel('Tiempo')
plt.grid(linestyle='dashed')
plt.legend(('Cinética', 'Potencial', 'Mécanica'), bbox_to_anchor=(1.05,0.8), loc=3, borderaxespad=0, prop={'size':9})

plt.show()
