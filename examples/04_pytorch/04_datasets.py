import torch
import numpy as np
from pathlib import Path


class OSEM2DDataSet(torch.utils.data.Dataset):

    def __init__(self,
                 basedir: str = '../data/OSEM_2D_5.00E+01',
                 seed: int = 1):
        self._basedir = Path(basedir)
        self._seed = seed

        self._dir_list = sorted(self._basedir.glob('???_???_???'))

    @property
    def basedir(self) -> Path:
        return self._basedir

    @property
    def seed(self) -> int:
        return self._seed

    @seed.setter
    def seed(self, value) -> None:
        self._seed = value

    @property
    def dir_list(self) -> list[Path]:
        return self._dir_list

    def __len__(self):
        return len(self.dir_list)

    def __getitem__(self, idx: int):
        odir = self.dir_list[idx]

        image = torch.from_numpy(np.expand_dims(np.load(odir / 'image.npy'),
                                                0))
        osem = torch.from_numpy(
            np.expand_dims(np.load(odir / f'osem_{self.seed:03}.npy'), 0))
        data = torch.from_numpy(
            np.load(odir / f'data_{self.seed:03}.npy').astype(np.int16))
        multiplicative_corrections = torch.from_numpy(
            np.load(odir / 'multiplicative_corrections.npy'))
        contamination = torch.from_numpy(np.load(odir / 'contamination.npy'))

        return (osem, data, multiplicative_corrections, contamination), image


if __name__ == '__main__':

    ds = OSEM2DDataSet()
    data_loader = torch.utils.data.DataLoader(ds, batch_size=10, shuffle=True)

    a, b = next(iter(data_loader))
