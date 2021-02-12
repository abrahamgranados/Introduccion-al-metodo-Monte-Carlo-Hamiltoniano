
import matplotlib.pyplot as plt
import numpy as np
from google.colab import files
from mpl_toolkits import mplot3d
# %matplotlib inline

"""#Capitulo 2

##2.2. Mecánica Clásica
"""

t = np.arange(0,10, 0.1) #tiempo
x=t*t
y=t+5
z=np.sin(t)

#puntos donde ira el vector 
tp=np.arange(3,10,3)
xp=tp*tp
yp=tp+5
zp=np.sin(tp) 

## derivadas 
dx=2*tp
dy=tp/tp*5
dz=np.cos(tp)

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.plot3D(x, y, z, c= 'teal')

ax.quiver(xp, yp, zp, dx, dy, dz, length=0.12, lw=1.5, color='r')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.legend(('trayectoria', 'vector de velocidades'), bbox_to_anchor=(1.05,0.8), loc=3, borderaxespad=0, prop={'size':9})
