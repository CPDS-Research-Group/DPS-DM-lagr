from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
import h5py


def load_data(
    *,
    file_list,
    rx0, rx1,
    batch_size,
    depth_size=None,
    class_cond=False,
    periodic_shift=False,
    deterministic=False,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param file_list: an ASCII file including paths of all fields in the dataset.
    :params rx0, rx1: min and max values for field normalization.
    :param batch_size: the batch size of each returned pair.
    :param depth_size: if specified, loads a subset of the depth dimension
                       with shape (N, C, depth_size, H, W). If None, loads the
                       full depth dimension with shape (N, C, D, H, W).
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. Not implemented.
    :param periodic_shift: if True, apply a random periodic shift in all
                           spatial dimensions to augment data.
    :param deterministic: if True, yield results in a deterministic order.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    with open(file_list, 'r') as f:
        field_paths = [line.strip() for line in f]

    dataset = Turb3dDataset(
        field_paths, rx0, rx1, depth_size, class_cond, periodic_shift,
        rng=np.random.default_rng(seed=rank), shard=rank, num_shards=size,
    )

    shuffle = False if deterministic else True
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=1, drop_last=True
    )

    while True:
        yield from loader


class Turb3dDataset(Dataset):
    def __init__(
        self,
        field_paths,
        rx0, rx1,
        depth_size,
        class_cond,
        periodic_shift,
        rng,
        shard=0,
        num_shards=1,
    ):
        super().__init__()
        self.local_paths = field_paths[shard:][::num_shards]
        self.rx0 = np.array(rx0, dtype=np.float32)
        self.rx1 = np.array(rx1, dtype=np.float32)
        self.depth_size = depth_size
        self.class_cond = class_cond
        self.periodic_shift = periodic_shift
        self.rng = rng

    def __len__(self):
        return len(self.local_paths)

    def __getitem__(self, idx):
        with h5py.File(self.local_paths[idx], 'r') as hf:
            data = np.stack((
                hf['/PS3D/vx'].astype(np.float32),
                hf['/PS3D/vy'].astype(np.float32),
                hf['/PS3D/vz'].astype(np.float32),
            ))
            data = 2*(data-self.rx0) / (self.rx1-self.rx0) - 1

        if self.periodic_shift:
            image_size, ndim = data.shape[1], data.ndim
            shift = self.rng.integers(-image_size//2, image_size//2, size=ndim-1)
            data = np.roll(data, shift, axis=np.arange(1, ndim))

        if self.depth_size is not None:
            idx0 = self.rng.integers(image_size-self.depth_size, endpoint=True)
            idx1 = idx0 + self.depth_size
            data = data[:, idx0:idx1].copy()

        out_dict = {}
        if self.class_cond:
            raise NotImplementedError()
        return data, out_dict
