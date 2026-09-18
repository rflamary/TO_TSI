# -*- coding: utf-8 -*-
"""
=================
Plot multiple EMD
=================

Shows how to compute multiple EMD and Sinkhorn with two differnt
ground metrics and plot their values for diffeent distributions.


"""
#%%
# Author: Remi Flamary 
#
# License: MIT License

import numpy as np
import matplotlib.pylab as pl
import ot
from ot.datasets import make_1D_gauss as gauss
import scipy.ndimage as ndimage

##############################################################################
# Generate data
# -------------

#%% parameters

n = 1000  # nb bins
n_target = 100  # nb target distributions

def gauss(n, m=0, s=1):
    """Generates a uniform distrbution around m"""
    x = np.arange(n, dtype=np.float64)
    w = np.abs(x - m) <  s-0.1
    return w / w.sum()

# bin positions
x = np.arange(n, dtype=np.float64)

lst_m = np.linspace(100, n-n/10, n_target).astype(np.int64)  # means of the target distributions

# Gaussian distributions
a = gauss(n, m=310, s=50)  # m= mean, s= std

B = np.zeros((n, n_target))

for i, m in enumerate(lst_m):
    B[:, i] = gauss(n, m=m, s=50)

# loss matrix and normalization
M = ot.dist(x.reshape((n, 1)), x.reshape((n, 1)), 'euclidean')
M /= M.max()
M2 = ot.dist(x.reshape((n, 1)), x.reshape((n, 1)), 'sqeuclidean')
M2 /= M2.max()

##############################################################################
# Plot data
# ---------
d_emd = ot.emd2(a, B, M)  # direct computation of EMD
d_emd2 = ot.emd2(a, B, M2)  # direct computation of EMD with loss M2

d_l1=np.sum(np.abs(B-a[:,None]),0)
d_l2=np.sum(np.abs(B-a[:,None])**2,0)
d_mmd=np.sum((ndimage.gaussian_filter(B, sigma=50, mode='wrap',axes=0)-ndimage.gaussian_filter(a[:,None], sigma=50, mode='wrap',axes=0))**2,0)



#% plot the distributions
from matplotlib import gridspec
step=15
k=3
B2=B[:,::step]
alpha=np.exp(k*np.linspace(0.1,1,B2.shape[1])-k)

x0 = 0*x

gs = gridspec.GridSpec(2, 2)

pl.figure(1,(12,3))
pl.clf()
pl.subplot(gs[0,0])
pl.plot(x, a, color='C3')
pl.fill_between(x, x0, a, color='C3', alpha=0.5)
pl.xticks(())
pl.yticks(())
pl.title('Distribution 1')
pl.subplot(gs[1,0])
for i,al in enumerate(alpha):
    pl.plot(x, B2[:,i], color='C0', alpha=al)
    pl.fill_between(x, x0, B2[:,i], color='C0', alpha=al*0.5)
pl.title('Distribution 2 (déplacement vers la droite)')
pl.xticks(())
pl.yticks(())

pl.subplot(gs[:,1])
#pl.plot(np.array(d_emd)*4, label='$W_1^1$')
pl.plot(np.array(d_emd2)*9, label='$W_2^2$')
pl.plot(d_l1, label='Variation totale ($l_1$)')
pl.plot(d_l2/d_l2.max()*1.5, label='Euclidienne ($l_2^2$)')
pl.plot(d_mmd/d_mmd.max()*1.5, label='MMD')
pl.title('Comparaison des divergences en fonction du déplacement')
pl.legend()
pl.xticks(())
pl.yticks(())
axl=pl.axis()
pl.tight_layout()

pl.savefig('../imgs/wass_plot.pdf',bbox_inches='tight')
pl.savefig('../imgs/wass_plot.png',bbox_inches='tight',dpi=300)
##############################################################################
# Compute EMD for the different losses
# ------------------------------------


#%%


# movie command line
#$ ffmpeg -framerate 10 -i 'wass_%03d.png' wass.mp4

# for i in range(n_target):
#     pl.close(1)
#     pl.figure(1,(8,3))
#     pl.clf()
#     pl.subplot(gs[0,0])
#     pl.plot(x, a, 'r')
#     pl.xticks(())
#     pl.yticks(())
#     pl.title('Source distribution')
#     pl.subplot(gs[1,0])
    
#     pl.plot(x, B[:,i],'b')
#     pl.title('Target distributions')
#     pl.xticks(())
#     pl.yticks(())
    
#     pl.subplot(gs[:,1])
#     pl.plot(np.array(d_emd[:i+1])*4, label='$W_1^1$')
#     pl.plot(np.array(d_emd2[:i+1])*9, label='$W_2^2$')
#     pl.plot(d_l1[:i+1], label='$l_1$ (TV)')
#     pl.plot(d_l2[:i+1]/d_l2.max()*1.5, label='$l_2$ (sq. eucl.)')
#     pl.plot(i,np.array(d_emd[i])*4,'+',color='C0')
#     pl.plot(i,np.array(d_emd2[i])*9,'+',color='C1')
#     pl.plot(i,d_l1[i],'+',color='C2')
#     pl.plot(i,d_l2[i]/d_l2.max()*1.5,'+',color='C3')    
    
#     pl.title('Divergences (scaled)')
#     pl.axis(axl)
#     pl.legend(loc=2)
#     pl.xticks(())
#     pl.yticks(())
    
#     pl.tight_layout()
    
#     pl.savefig('../imgs/wass/wass_{:03d}.png'.format(i),bbox_inches='tight',dpi=200)    

