import os
import numpy as np
from numpy.lib.format import open_memmap


def run(input_path):
    sets = {
        'test'

    }

    datasets = {
        input_path
    }

    parts = {
        'joint', 'bone'
    }
    from tqdm import tqdm

    for dataset in datasets:
        for set in sets:
            for part in parts:
                print(dataset, set, part)
                data = np.load('{}/{}_data_{}.npy'.format(dataset, set, part))
                N, C, T, V, M = data.shape
                print(data.shape)
                fp_sp = open_memmap(
                    '{}/{}_data_{}_motion.npy'.format(dataset, set, part),
                    dtype='float32',
                    mode='w+',
                    shape=(N, C, T, V, M))
                for t in tqdm(range(T - 1)):
                    fp_sp[:, :, t, :, :] = data[:, :, t + 1, :, :] - data[:, :, t, :, :]
                fp_sp[:, :, T - 1, :, :] = 0
