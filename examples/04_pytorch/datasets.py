import torch
import numpy as np
from pathlib import Path


class OSEM2DDataSet(torch.utils.data.Dataset):

    def __init__(self,
                 basedir: str,
                 seed: int = 1,
                 normalization_quantile: float = 0.99,
                 verbose: bool = False,
                 search_pattern: str = '???_???_???') -> None:
        """simulated 2D PET data

        Parameters
        ----------
        basedir : str, optional
            base directory containing all simulated 2D data sets, by default '../data/OSEM_2D_5.00E+01'
        seed : int, optional
            seed used for random generator, by default 1
        normalization_quantile : float, optional
            quantile used to normalize the OSEM and true image, by default 0.99
        verbose: bool, optional
            print verbose output, by default False
        search_pattern: str, optional
            search pattern used to find data sets, default '???_???_???'
        """
        self._basedir = Path(basedir)
        self._seed = seed
        self._normalization_quantile = normalization_quantile
        self._verbose = verbose

        self._dir_list = sorted(self._basedir.glob(search_pattern))

        self._target_image_name = 'image.npz'

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
    def verbose(self) -> bool:
        return self._verbose

    @verbose.setter
    def verbose(self, value) -> None:
        self._verbose = value

    @property
    def dir_list(self) -> list[Path]:
        return self._dir_list

    @property
    def normalization_quantile(self) -> float:
        return self._normalization_quantile

    @property
    def input_image_name(self) -> str:
        return f'osem_{self.seed:03}.npz'

    @property
    def data_name(self) -> str:
        return f'data_{self.seed:03}.npz'

    @property
    def target_image_name(self) -> str:
        return self._target_image_name

    @target_image_name.setter
    def target_image_name(self, value: str) -> None:
        self._target_image_name = value

    def __len__(self) -> int:
        return len(self.dir_list)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor]:
        """load sample of OSEM 2D data set

        Parameters
        ----------
        idx : int
            sample number

        Returns
        -------
        tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
          (normalized OSEM recon, noisy emission data (sinogram), multiplicative_corrections sinogram, 
           contamination sinogram, image normalization factor, 
           normalized true image
        """
        odir = self.dir_list[idx]

        if self.verbose:
            print(odir)

        image = torch.from_numpy(
            np.expand_dims(np.load(odir / self.target_image_name)['arr_0'], 0))
        osem = torch.from_numpy(
            np.expand_dims(np.load(odir / self.input_image_name)['arr_0'], 0))
        adjoint_ones = torch.from_numpy(
            np.load(odir / 'adjoint_ones.npz')['arr_0'])
        data = torch.from_numpy(
            np.load(odir / self.data_name)['arr_0'].astype(np.int16))
        multiplicative_corrections = torch.from_numpy(
            np.load(odir / 'multiplicative_corrections.npz')['arr_0'])
        contamination = torch.from_numpy(
            np.load(odir / 'contamination.npz')['arr_0'])

        # normalize the OSEM and true image
        norm = torch.quantile(osem, self.normalization_quantile)

        osem /= norm
        image /= norm

        return osem, data, multiplicative_corrections, contamination, adjoint_ones, norm, image


if __name__ == '__main__':

    ds = OSEM2DDataSet()
    data_loader = torch.utils.data.DataLoader(ds, batch_size=10, shuffle=True)

    a = next(iter(data_loader))
