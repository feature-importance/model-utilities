import io
import os
import pickle

import h5py
from PIL import Image
from torchvision.datasets import VisionDataset


class ImageNetHDF5(VisionDataset):
    def __init__(self, root, cache_size=None, transform=None, classes=None, lazy=True):
        super(ImageNetHDF5, self).__init__(root, transform=transform, target_transform=None)

        if cache_size is not None:
            print("caching has been removed and is no longer supported.")
        if not lazy:
            print("lazy has been removed and is no longer supported.")

        self.dest = pickle.load(open(os.path.join(root, 'dest.p'), 'rb'))

        targets = sorted(list(filter(lambda f: '.hdf5' in f, os.listdir(root))))
        if classes:
            targets = sorted(list(filter(lambda f: f[:-5] in classes, targets)))

        self.targets = {f[:-5]: i for i, f in enumerate(targets)}

        if classes:
            newdest = []
            for idx in range(len(self)):
                dest, i = self.dest[idx]
                if dest in self.targets:
                    newdest.append((dest, i))
            self.dest = newdest

        self._open_files = {}
        self._owner_pid = None

    def _get_dataset(self, name):
        pid = os.getpid()

        if self._owner_pid != pid:
            self.close()
            self._owner_pid = pid

        if name not in self._open_files:
            handle = h5py.File(os.path.join(self.root, name + ".hdf5"),"r")
            try:
                dataset = handle["data"]
            except Exception:
                handle.close()
                raise

            self._open_files[name] = (handle, dataset)

        return self._open_files[name][1]

    def load(self, name, index):
        return self._get_dataset(name)[index]

    def close(self):
        handles = getattr(self, "_open_files", {})
        for handle, _ in handles.values():
            try:
                handle.close()
            except Exception:
                pass
        handles.clear()
        self._owner_pid = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        self.close()

    def __getstate__(self):
        # don't pickle file handles
        state = self.__dict__.copy()
        state["_open_files"] = {}
        state["_owner_pid"] = None
        return state

    # def load(self, file, i):
    #     with h5py.File(os.path.join(self.root, file + '.hdf5'), 'r') as f:
    #         return f['data'][i]

    def __getitem__(self, index):
        dest, i = self.dest[index]

        sample = self.load(dest, i)

        sample = Image.open(io.BytesIO(sample))
        sample = sample.convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, self.targets[dest]

    def __len__(self):
        return len(self.dest)
