## 4 Subplots

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

#
def build_subplot():
    """
    """
    titles = ['$\lambda = 0.5$',
              '$\lambda = 1.0$',
              '$\lambda = 1.5$',
              '$\lambda = 2.0$']
    fig_letter = ['(a)', '(b)', '(c)', '(d)']

    nelx = 5
    nely = 5

    A = dict()
    A[0] = np.random.rand(5, 5)
    A[1] = np.random.rand(5, 5)
    A[2] = np.random.rand(5, 5)
    A[3] = np.random.rand(5, 5)
    
    fig, axs = plt.subplots(1, 4, figsize=(10, 3))
    plt.subplots_adjust(wspace=0.4)

    for n in range(4):
        axs[n].imshow(A[n], cmap=cm.gray, extent=[0, nelx, 0, nely])
        axs[n].set_title(titles[n].capitalize())
        
        axs[n].set_xlabel(fig_letter[n], labelpad=10, fontsize=14)
        axs[n].set_xticks(np.arange(0, nelx+1, step=1))
        axs[n].set_yticks(np.arange(0, nely+1, step=1))
        axs[n].grid(True)

    plt.show() 


## BUCKLING ERROR
def validate_buckling():
    ndof = [44, 126, 248, 410, 612, 854, 1136]
    error = [49.39, 11.87, 4.89, 2.45, 1.32, 0.70, 0.33]
    plt.plot(ndof, error,'-ko')

    plt.xlabel('$N.^o$ de graus de liberdade', fontsize=12)
    plt.ylabel('Erro [%]', fontsize=12)
    plt.xticks(np.arange(0, 1400, step=200))
    plt.xlim((0, 1200))
    plt.yticks(np.arange(0, 60, step=10))
    plt.ylim((-5, 50))
    plt.grid()
    plt.show()