import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np


x = np.arange(101)
x = 1e-2 * x

fig = plt.figure()
ax = fig.add_subplot(111)

# Grids and Thicks
ax.set_xticks(np.arange(0, 1.2, step=0.2))
ax.set_yticks(np.arange(0, 1.2, step=0.2))

ax.grid(True)

ax.axis([0, 1, 0, 1])
#ax.axis('scale')
ax.set_xlabel('Infill density ($x_e$)', fontsize=14)
ax.set_ylabel('Penalization ($p(xe)$)', fontsize=14)

ax.plot(x, x, linestyle='dashed', color='b', label='$p_e = x_e$')
ax.plot(x, x**3, linestyle='dashdot',  color='r', label='$p_e = x_e^3$')
ax.plot(x, x**1.484, color='k', label='$p_e = x_e^{1.484}$')

ax.legend(fontsize=14)


plt.tight_layout()
plt.show()