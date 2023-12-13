# coding along with mtrf docs to familiarize with its usage: 
# https://mtrfpy.readthedocs.io/en/latest/basics.html

# %%
from mtrf.model import load_sample_data

X, R, fs = load_sample_data(n_segments=10, normalize=True)
# Is 10 the number of stimuli? or trials?
# NOTE: X and R are lists, each containing stim and responses for one stim
# %% 
from mtrf.model import TRF
fwd_trf = TRF(direction=1)

tmin, tmax = 0.0, 0.4 # time lags

fwd_trf.train(X, R, fs, tmin, tmax, regularization=1000)

 # %% 
R_hat, r_fwd, mse_fwd = fwd_trf.predict(X, R)
print(f"Correlation: {r_fwd}")

# %%
from mtrf.stats import cross_validate
cv_r_fwd, mse_fwd = cross_validate(fwd_trf, X, R)
print(f"cross-validated correltaion: {cv_r_fwd}")
# preforms k-fold cross validation; 
# default number of folds (k) is "-1" aka LOO
# seems 10 is indeed the number of stimuli (or trials....?)
# didn't need to do mean/round since r_fwd was only one number?

# %% Backward Model
# note that stimuli here are actually spectrograms
# so can get "envelope" from averaging across spectral bands
#  
env = [x.mean(axis=-1) for x in X]
bwd_trf = TRF(direction=-1)
bwd_trf.train(env, R, fs, tmin, tmax, regularization=1000)
r_bwd, mse_bwd, = cross_validate(bwd_trf, env, R)
print(f" bwd Correlation: {r_bwd}")

# %% visualization
# note: plot acts on the weight matrix, which has dims [input x lags x outputs]
# thus need to average across a dim or select an input/"feature" or output/"channel"
# to look at
from matplotlib import pyplot as plt
fig, ax = plt.subplots(2)

fwd_trf.plot(feature=6, axes=ax[0], show=False)
fwd_trf.plot(channel='gfp', axes=ax[1], kind='image', show=False)
# NOTE: why show=False?
# NOTE: how are STRF and global field power equivalent?
plt.tight_layout()
plt.show()

# %% Optimization
import numpy as np
# NOTE: passing list of reg params to TRF.train will induce cross val
trf = TRF()
# already have sample data - although in example
# here they didn't normalize and we have normalized data
regularization = np.logspace(-1, 6, 20)
corr, err = trf.train(
    X, R, fs, tmin, tmax, regularization, k=-1
)
fig, ax1 = plt.subplots()
ax2 = ax1.twinx() # cool
ax1.semilogx(regularization, corr, color='c')
ax2.semilogx(regularization, err, color='m')
ax1.set(xlabel="\Lambda", ylabel= "Correlation Coefficient")
ax2.set(ylabel="MSE")
ax1.axvline(regularization[np.argmin(err)], linestyle='--', color='k')
plt.show()