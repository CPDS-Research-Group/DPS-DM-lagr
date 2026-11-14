import h5py
import numpy as np

with h5py.File('Lagr_u3c_diffusion_splits.h5', 'r') as hf:
    rx0 = np.array(hf['min'])
    rx1 = np.array(hf['max'])
    x_train = np.array(hf['train'])
    x_test  = np.array(hf['test'])

rng = np.random.default_rng(seed=0)
idx = rng.permutation(x_train.shape[0])
x_train = x_train[idx].copy()

with h5py.File('Lagr_u3c_diffusion_splits_tfg.h5', 'w') as hf:
    hf.create_dataset('min', data=rx0)
    hf.create_dataset('max', data=rx1)
    hf.create_dataset('train', data=x_train[:-32768])
    hf.create_dataset('valid', data=x_train[-32768:])
    hf.create_dataset('test',  data=x_test)
