import matplotlib.pyplot as plt
import numpy as np
from google.colab import files
# %matplotlib inline

t = np.arange(0,2*(np.pi), 0.05)
plt.text(0.05, -0.18, r'A', fontsize=15)   # A
plt.text(2*np.pi-0.05, 0.10, r'B', fontsize=15)  # B
plt.text(4, 0.05, r'v',  fontsize=15)  #Camino V
plt.text(1.5, 0.79, r'$w$',  fontsize=15)  #camino W
plt.plot(t,np.sin(t),  'darkblue',linestyle='--')  #
plt.plot([0, 2*np.pi], [0, 0], 'darkblue',linestyle='--')
plt.plot([0], [0], 'maroon', marker=".", markersize=20)   
plt.plot([2*np.pi], [0], 'maroon', marker=".", markersize=20)
plt.axis('off')
plt.grid(linestyle='dashed')
#plt.savefig("conservativas.png",bbox_inches='tight',dpi=300)
#files.download("conservativas.png")
