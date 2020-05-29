## 4 Subplots

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

#
def build_subplot():
    """
    """
    titles = ['P$_c$ = 1.0',
              'P$_c$ = 1.5',
              'P$_c$ = 2.0',
              'P$_c$ = 2.5']
    fig_letter = ['(a)', '(b)', '(c)', '(d)']

    nelx = 60
    nely = 30

    A = dict()
    A[0] = np.loadtxt('new_results_txt\BColumnBuckling\B1_final_Column 60x30[p=3, buckling=0.0, vol=0.3]_densities.dat]_densities.dat')
    A[1] = np.loadtxt('new_results_txt\BColumnBuckling\B2_final_Column 60x30[p=3, buckling=0.080, vol=0.3]_densities.dat]_densities.dat')
    A[2] = np.loadtxt('new_results_txt\BColumnBuckling\B3_final_Column 60x30[p=3, buckling=0.106, vol=0.3]_densities.dat]_densities.dat')
    A[3] = np.loadtxt('new_results_txt\BColumnBuckling\B4_final_Column 60x30[p=3, buckling=0.133, vol=0.3]_densities.dat]_densities.dat')

    print(A[0].shape)
    
    axs = dict()

    #fig, ((axs[0], axs[1]), (axs[2], axs[3])) = plt.subplots(2, 2)#figsize=(4, 2))
    fig, ((axs[0], axs[1], axs[2], axs[3])) = plt.subplots(1, 4, figsize=(7,3.5))
    #plt.subplots_adjust(wspace=0.05)
    #plt.subplots_adjust(hspace=0.4)

    for n in range(4):
        axs[n].imshow(-np.flip(np.transpose(A[n].reshape((nely, nelx))),0),
                      cmap=cm.gray,
                      extent=[0, nely, 0, nelx])

        axs[n].set_title(titles[n].capitalize(), fontsize=12)
        
        axs[n].set_xlabel(fig_letter[n], labelpad=10, fontsize=12)
        axs[n].set_xticks(np.arange(0, nely+1, step=30))
        axs[n].set_yticks(np.arange(0, nelx+1, step=30))

        axs[n].set_xticks(np.arange(0, nely+1, step=5), minor=True)
        axs[n].set_yticks(np.arange(0, nelx+1, step=5), minor=True)
        
        axs[n].grid(b=True, which='major', alpha=0.5)
        axs[n].grid(b=True, which='minor', alpha=0.2, color='#666666')

    plt.tight_layout()
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
    plt.yticks(np.arange(0, 60, step=30))
    plt.ylim((-5, 50))
    plt.grid()
    plt.show()


print('is it?')
build_subplot()
print("EOF")
print('...')


