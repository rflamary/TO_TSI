# Author: Remi Flamary 
#
# License: MIT License

#%% 

import numpy as np
import ot
import matplotlib.pyplot as pl


#%%
# Create two distributions

n=20
a = np.ones((n,)) / n  # uniform distribution on source
b = np.ones((n,)) / n  # uniform distribution on target

sigma = 0.1

np.random.seed(42)

#angle_s = np.random.rand(n) * 2 * np.pi
angle_s = np.linspace(0, 2 * np.pi, n, endpoint=False) + np.random.rand(n) * 0.1
xs = np.vstack((np.cos(angle_s), np.sin(angle_s))).T + sigma * np.random.randn(n, 2)

#angle_t = np.random.rand(n) * 2 * np.pi
angle_t = np.linspace(0, 2 * np.pi, n, endpoint=False) + np.random.rand(n) * 0.5
xt = 3*np.vstack((np.cos(angle_t), np.sin(angle_t))).T + sigma * np.random.randn(n, 2)

alpha=0.7

pl.figure(1, figsize=(5, 5))
pl.scatter(xs[:, 0], xs[:, 1], color='C3', label='Source', alpha=alpha)
pl.scatter(xt[:, 0], xt[:, 1], color='C0', label='Target', alpha=alpha)
pl.legend()
pl.title('Problème de Monge')

#%%

# cost matrix
M = ot.dist(xs, xt)

res = ot.solve(M, a, b)
plan = res.plan

# plot solution
arrow_scale = 1
pl.figure(2, figsize=(5, 5))
pl.scatter(xs[:, 0], xs[:, 1], color='C3', alpha=alpha)
pl.scatter(xt[:, 0], xt[:, 1], color='C0', alpha=alpha)
#pl.plot([],[], color='k', label='$T$',alpha=0.5)
pl.scatter( [],[], c='k',marker=r'$\rightarrow$',s=100, label='App. de Monge',alpha=0.5)

for i in range(n):
    for j in range(n):
        if plan[i, j] > 1/31:
            # plot arrow from xs[i] to xt[j]
            pl.arrow(xs[i, 0], xs[i, 1], arrow_scale*(xt[j, 0] - xs[i, 0]), arrow_scale*(xt[j, 1] - xs[i, 1]), color='k', head_width=0.2,alpha=0.5, length_includes_head=True,)

pl.legend()
pl.title('Application de Monge')

plan_sparse = plan
plan_sparse[plan_sparse < 1/31] = np.nan

pl.figure(3, figsize=(5, 5))
#pl.imshow(M, interpolation='nearest')
pl.pcolor(M, cmap='Blues', alpha=1)
pl.pcolor(plan_sparse, cmap='gray', alpha=1, vmin=0,vmax=0.1)
pl.scatter([],[], c='k',marker=r's',s=100, label='Solution $\sigma$',alpha=0.5)
pl.title('Matrice de coût + solution $\sigma$')
pl.xticks(np.arange(n,step=5)+0.5, np.arange(n,step=5))
pl.yticks(np.arange(n,step=5)+0.5, np.arange(n,step=5))
pl.xlabel('j')
pl.ylabel('i')
pl.legend()


#%% plot monge exemple

pl.figure(1, figsize=(14, 4))

pl.subplot(1, 3, 1)
pl.scatter(xs[:, 0], xs[:, 1], color='C3', label='$x_i$', alpha=alpha)
pl.scatter(xt[:, 0], xt[:, 1], color='C0', label='$y_j$', alpha=alpha)
pl.legend()
pl.title('Problème de Monge')


pl.subplot(1, 3, 2)
pl.pcolor(M, cmap='Greens', alpha=1)
pl.pcolor(plan_sparse, cmap='Greys', alpha=1, vmin=0,vmax=1/n*1.5)
pl.scatter([],[], c='k',marker=r's',s=100, label=r'Solution $\sigma$',alpha=0.5)
pl.title(r'Matrice de coût $c(x_i,y_j)$ et solution $\sigma$')
pl.xticks(np.arange(n,step=5)+0.5, np.arange(n,step=5))
pl.yticks(np.arange(n,step=5)+0.5, np.arange(n,step=5))
pl.xlabel('j')
pl.ylabel('i')
pl.legend(loc='upper left')

pl.subplot(1, 3, 3)
pl.scatter(xs[:, 0], xs[:, 1], color='C3', alpha=alpha)
pl.scatter(xt[:, 0], xt[:, 1], color='C0', alpha=alpha)
pl.scatter( [],[], c='k',marker=r'$\rightarrow$',s=100, label='App. de Monge',alpha=0.5)

for i in range(n):
    for j in range(n):
        if plan[i, j] > 1/31:
            # plot arrow from xs[i] to xt[j]
            pl.arrow(xs[i, 0], xs[i, 1], arrow_scale*(xt[j, 0] - xs[i, 0]), arrow_scale*(xt[j, 1] - xs[i, 1]), color='k', head_width=0.2,alpha=0.5, length_includes_head=True,)
pl.legend()
pl.title('Application de Monge $T$')

pl.savefig('../imgs/monge_example.png', dpi=300, bbox_inches='tight')
pl.savefig('../imgs/monge_example.pdf', bbox_inches='tight')


#%% new data pour katoroivitch

a2 = a+0.1*np.random.rand(n)
a2 = a2/np.sum(a2)

b2 = b+0.1*np.random.rand(n)
b2 = b2/np.sum(b2)

# solution + visu

res2 = ot.solve(M, a2, b2)
plan2 = res2.plan

plan2_sparse = plan2
plan2_sparse[plan2_sparse < 1e-5] = np.nan

pl.figure(1, figsize=(14, 4))
scale = 50*n
pl.subplot(1, 3, 1)
pl.scatter(xs[:, 0], xs[:, 1], s=scale*a2, color='C3', label='$x_i$', alpha=alpha)
pl.scatter(xt[:, 0], xt[:, 1], s=scale*b2, color='C0', label='$y_j$', alpha=alpha)
pl.legend()
pl.title('Problème de Kantorovitch')


pl.subplot(1, 3, 2)
pl.pcolor(M, cmap='Greens', alpha=1)
pl.pcolor(plan2_sparse, cmap='Greys', alpha=1, vmin=-0.05,vmax=1/n*1.5)
pl.scatter([],[], c='k',marker=r's',s=100, label=r'Solution $P_{ij}>0$',alpha=0.5)
pl.title(r'Matrice de coût $c(x_i,y_j)$ et solution $P$')
pl.xticks(np.arange(n,step=5)+0.5, np.arange(n,step=5))
pl.yticks(np.arange(n,step=5)+0.5, np.arange(n,step=5))
pl.xlabel('j')
pl.ylabel('i')
pl.legend(loc='upper left')

pl.subplot(1, 3, 3)
pl.scatter(xs[:, 0], xs[:, 1], s=scale*a2, color='C3', alpha=alpha)
pl.scatter(xt[:, 0], xt[:, 1], s=scale*b2, color='C0', alpha=alpha)
pl.plot([],[], color='k', label='Plan optimal $P_{i,j}>0$',alpha=0.7)
for i in range(n):
    for j in range(n):
        if plan2[i, j] > 0:
            # plot line from xs[i] to xt[j]
            pl.plot([xs[i, 0], xt[j, 0]], [xs[i, 1], xt[j, 1]], color='k', alpha=0.7*plan2[i, j]*n, linewidth=2)
pl.legend()
pl.title('Plan optimal $P$')

pl.savefig('../imgs/kanto_example.png', dpi=300, bbox_inches='tight')
pl.savefig('../imgs/kanto_example.pdf', bbox_inches='tight')

#%% plot barycenters

# bary 1
t1 = 0.2

xbary =np.array([(1-t1)*xs[i] + t1*xt[j] for i in range(n) for j in range(n) if plan2[i, j] > 0])
abary1 = np.array([plan2[i, j] for i in range(n) for j in range(n) if plan2[i, j] > 0])

# bary 2
t2 = 0.5
xbary2 =np.array([(1-t2)*xs[i] + t2*xt[j] for i in range(n) for j in range(n) if plan2[i, j] > 0])
abary2 = np.array([plan2[i, j] for i in range(n) for j in range(n) if plan2[i, j] > 0])

pl.figure(1, figsize=(14, 4))

pl.subplot(1, 3, 1)
pl.scatter(xs[:, 0], xs[:, 1], s=scale*a2, color='C3', label='$x_i$', alpha=alpha)
pl.scatter(xt[:, 0], xt[:, 1], s=scale*b2, color='C0', label='$y_j$', alpha=alpha)
for i in range(n):
    for j in range(n):
        if plan2[i, j] > 0:
            pl.plot([xs[i, 0], xt[j, 0]], [xs[i, 1], xt[j, 1]], color='k', alpha=0.7*plan2[i, j]*n, linewidth=2)    
pl.legend()
pl.title('Problème et plan optimal $P$')

pl.subplot(1, 3, 2)
pl.scatter(xs[:, 0], xs[:, 1], s=scale*a2, color='C3', alpha=alpha*0.7)
pl.scatter(xt[:, 0], xt[:, 1], s=scale*b2, color='C0',  alpha=alpha*0.7)
for i in range(n):
    for j in range(n):
        if plan2[i, j] > 0:
            pl.plot([xs[i, 0], xt[j, 0]], [xs[i, 1], xt[j, 1]], color='k', alpha=0.2*plan2[i, j]*n, linewidth=2) 
pl.scatter(xbary[:, 0], xbary[:, 1], s=scale*abary1, color='C2', label='$z_k$', alpha=alpha)
pl.legend()
pl.title('Barycentre pour $t=0.2$')

pl.subplot(1, 3, 3)
pl.scatter(xs[:, 0], xs[:, 1], s=scale*a2, color='C3', alpha=alpha*0.7)
pl.scatter(xt[:, 0], xt[:, 1], s=scale*b2, color='C0',  alpha=alpha*0.7)
for i in range(n):
    for j in range(n):
        if plan2[i, j] > 0:
            pl.plot([xs[i, 0], xt[j, 0]], [xs[i, 1], xt[j, 1]], color='k', alpha=0.2*plan2[i, j]*n, linewidth=2)
pl.scatter(xbary2[:, 0], xbary2[:, 1], s=scale*abary2, color='C2', label='$z_k$', alpha=alpha)
pl.legend()
pl.title('Barycentre pour $t=0.5$')

pl.savefig('../imgs/bary_example.png', dpi=300, bbox_inches='tight')
pl.savefig('../imgs/bary_example.pdf', bbox_inches='tight')


#%% illustrer me transport entropique


plan3 = ot.solve(M, a2, b2, reg=1).plan

plan4 = ot.solve(M, a2, b2, reg=0.2).plan

pl.figure(1, figsize=(14, 4))


pl.subplot(1, 3, 1)
pl.scatter(xs[:, 0], xs[:, 1], s=scale*a2, color='C3', label='$x_i$', alpha=alpha)
pl.scatter(xt[:, 0], xt[:, 1], s=scale*b2, color='C0', label='$y_j$', alpha=alpha)
for i in range(n):
    for j in range(n):
        if plan2[i, j] > 0.01/n:
            pl.plot([xs[i, 0], xt[j, 0]], [xs[i, 1], xt[j, 1]], color='k', alpha=0.7*plan2[i, j]*n+0.1, linewidth=2)
pl.title(r'Plan optimal exact')

pl.subplot(1, 3, 2)
pl.scatter(xs[:, 0], xs[:, 1], s=scale*a2, color='C3', label='$x_i$', alpha=alpha)
pl.scatter(xt[:, 0], xt[:, 1], s=scale*b2, color='C0', label='$y_j$', alpha=alpha)
for i in range(n):
    for j in range(n):
        if plan4[i, j] > 0.01/n:
            pl.plot([xs[i, 0], xt[j, 0]], [xs[i, 1], xt[j, 1]], color='k', alpha=0.7*plan4[i, j]*n+0.1, linewidth=2)
pl.title(r'Plan optimal entropique $\epsilon=0.2$')

pl.subplot(1, 3, 3)
pl.scatter(xs[:, 0], xs[:, 1], s=scale*a2, color='C3', label='$x_i$', alpha=alpha)
pl.scatter(xt[:, 0], xt[:, 1], s=scale*b2, color='C0', label='$y_j$', alpha=alpha)
for i in range(n):
    for j in range(n):
        if plan3[i, j] > 0.01/n:
            pl.plot([xs[i, 0], xt[j, 0]], [xs[i, 1], xt[j, 1]], color='k', alpha=0.7*plan3[i, j]*n+0.1, linewidth=2)
pl.title(r'Plan optimal entropique $\epsilon=1$')

pl.savefig('../imgs/entropic_example.png', dpi=300, bbox_inches='tight')
pl.savefig('../imgs/entropic_example.pdf', bbox_inches='tight')