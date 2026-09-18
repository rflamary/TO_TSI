# -*- coding: utf-8 -*-
"""
"""

# Author: Remi Flamary 
#
# License: MIT License

#%% 
import numpy as np
import matplotlib.pylab as pl
import ot
import os

import numpy as np
import ot

plt = pl
rng = np.random.RandomState(42)


def im2mat(img):
    """Converts an image to matrix (one pixel per line)"""
    return img.reshape((img.shape[0] * img.shape[1], img.shape[2]))


def mat2im(X, shape):
    """Converts back a matrix to an image"""
    return X.reshape(shape)


def minmax(img):
    return np.clip(img, 0, 1)

# %%

data_path = 'data/'

I1 = pl.imread(os.path.join(data_path, 'ocean_day.jpg')).astype(np.float64) / 256
I2 = pl.imread(os.path.join(data_path, 'ocean_sunset.jpg')).astype(np.float64) / 256

X1 = im2mat(I1)
X2 = im2mat(I2)

# %%

plt.figure(1, figsize=(6.4, 3))

plt.subplot(1, 2, 1)
plt.imshow(I1)
plt.axis('off')
plt.title('Image 1')

plt.subplot(1, 2, 2)
plt.imshow(I2)
plt.axis('off')
plt.title('Image 2')
# %% subsample pixels for visualization

# training samples
nb = 500
idx1 = rng.randint(X1.shape[0], size=(nb,))
idx2 = rng.randint(X2.shape[0], size=(nb+1,))

Xs = X1[idx1, :]
Xt = X2[idx2, :]


fig =plt.figure( 2,figsize=(6.4, 5))
pl.clf()
ax = fig.add_subplot(1,2,1,projection='3d')

ax.scatter(Xs[:, 0], Xs[:, 1], Xs[:, 2], c=Xs)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_zlim([0, 1])
ax.set_xlabel('Rouge')
ax.set_ylabel('Vert')
ax.set_zlabel('Bleu')
ax.set_title('Image 1')

ax = fig.add_subplot(1,2,2,projection='3d')

ax.scatter(Xt[:, 0], Xt[:, 1], Xt[:, 2], c=Xt)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_zlim([0, 1])

ax.set_xlabel('Rouge')
ax.set_ylabel('Vert')
ax.set_zlabel('Bleu')
ax.set_title('Image 2')


#%% train model

id_pix=500

adapt = ot.da.EMDTransport()

adapt.fit(Xs=Xs,Xt=Xt)

Xst=minmax(adapt.transform(Xs))
X1t=minmax(adapt.transform(X1))

Xtt=minmax(adapt.inverse_transform(Xt=Xt))
X2t=minmax(adapt.inverse_transform(Xt=X2))

I1t = mat2im(X1t, I1.shape)



#%%
fig =plt.figure(5,figsize=(13, 6))
pl.clf()


axi1=pl.subplot(2,3,4)
plt.imshow(I1)
plt.axis('off')
plt.title('Image 1')

axi2=pl.subplot(2,3,6)
plt.imshow(I2)
plt.axis('off')
plt.title('Image 2')

axd1 = fig.add_subplot(2,3,1,projection='3d')

axd1.scatter(Xs[:id_pix+1, 0], Xs[:id_pix+1, 1], Xs[:id_pix+1, 2], c=Xs[:id_pix+1,:],alpha=0.8)
axd1.set_xlim([0, 1])
axd1.set_ylim([0, 1])
axd1.set_zlim([0, 1])
axd1.set_xlabel('Rouge', labelpad=-13)
axd1.set_ylabel('Vert', labelpad=-13)
axd1.set_zlabel('Bleu', labelpad=-13)
axd1.set_title('Distribution de pixels 1')
axd1.set_xticklabels([])
axd1.set_yticklabels([])
axd1.set_zticklabels([])


axd2 = fig.add_subplot(2,3,3,projection='3d')

axd2.scatter(Xt[:id_pix+1, 0], Xt[:id_pix+1, 1], Xt[:id_pix+1, 2], c=Xt[:id_pix+1,:],alpha=0.8)
axd2.set_xlim([0, 1])
axd2.set_ylim([0, 1])
axd2.set_zlim([0, 1])
axd2.set_xlabel('Rouge', labelpad=-13)
axd2.set_ylabel('Vert', labelpad=-13)
axd2.set_zlabel('Bleu', labelpad=-13)
axd2.set_xticklabels([])
axd2.set_yticklabels([])
axd2.set_zticklabels([])
axd2.set_title('Distribution de pixels 2')

axi1=pl.subplot(2,3,4)
plt.imshow(I1)
plt.axis('off')
plt.title('Image 1')

axi2=pl.subplot(2,3,6)
plt.imshow(I2)
plt.axis('off')
plt.title('Image 2')


axd3 = fig.add_subplot(2,3,2,projection='3d')
nb = 40
axd3.scatter(Xs[:id_pix+1, 0], Xs[:id_pix+1, 1], Xs[:id_pix+1, 2], c=Xs[:id_pix+1,:],alpha=0.8)
axd3.scatter(Xt[:id_pix+1, 0], Xt[:id_pix+1, 1], Xt[:id_pix+1, 2], c=Xt[:id_pix+1,:],alpha=0.8,zorder=-5)
for i in range(nb):
    axd3.plot([Xs[i,0],Xst[i,0]],[Xs[i,1],Xst[i,1]],[Xs[i,2],Xst[i,2]],'k',alpha=0.5,zorder=1)
axd3.set_xlabel('Rouge', labelpad=-13)
axd3.set_ylabel('Vert', labelpad=-13)
axd3.set_zlabel('Bleu', labelpad=-13)
axd3.set_xticklabels([])
axd3.set_yticklabels([])
axd3.set_zticklabels([])
axd3.set_title('Transport entre les distributions')


axi3=pl.subplot(2,3,5)
axi3.imshow(I1t)
axi3.axis('off')
axi3.set_title('Image 1 adaptée')
#pl.savefig('../imgs/images_3d_disrib.pdf',bbox_inches='tight')
pl.savefig('../imgs/images_color_adapt.pdf',bbox_inches='tight')
pl.savefig('../imgs/images_color_adapt.png',dpi=300,bbox_inches='tight')

