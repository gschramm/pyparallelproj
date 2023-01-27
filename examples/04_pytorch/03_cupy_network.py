"""minimal example that shows how to setup a cascaded model with layers combining
   (cupy) projections and 3D convolutions
"""

import collections
import torch
import dill

import pyparallelproj.petprojectors as petprojectors

from layers import LinearSubsetForwardLayer, LinearSubsetAdjointLayer
from datasets import OSEM2DDataSet


class MyModel(torch.nn.Module):
    """dummy cascaded model that includes layers combining projections and convolutions"""

    def __init__(
        self,
        proj: petprojectors.PETJosephProjector,
        neural_net: torch.nn.Module,
    ) -> None:

        super(MyModel, self).__init__()
        self._proj = proj
        self._neural_net = neural_net
        self._neural_net_weight = torch.nn.Parameter(torch.tensor(1.0))

        self._pet_fwd_subset = LinearSubsetForwardLayer.apply
        self._pet_adjoint_subset = LinearSubsetAdjointLayer.apply

    def _data_fidelity_subset_update(self, osem: torch.Tensor,
                                     data: torch.Tensor,
                                     multiplicative_corrections: torch.Tensor,
                                     contamination: torch.Tensor,
                                     adjoint_ones: torch.Tensor,
                                     norm: torch.Tensor,
                                     subset: int) -> torch.Tensor:
        """subset data fidelity update for batch of images

        Parameters
        ----------
        osem : torch.Tensor
            batch of normalized OSEM images, shape (batch_size,0,n0,n1,1)
        data : torch.Tensor
            batch of (subset chunked) emission sinograms, shape (batch_size, num_subsets, num_subset_lors, num_tof_bins)
        multiplicative_corrections : torch.Tensor
            batch of (subset chunked) mult. correction sinograms, shape (batch_size, num_subsets, num_subset_lors, 1)
        contamination : torch.Tensor
            batch of (subset chunked) contamination sinograms, shape (batch_size, num_subsets, num_subset_lors, num_tof_bins)
        adjoint_ones : torch.Tensor
            batch of adjoint ones (subset sensitivity image), shape (batch_size, num_subsets, n0, n1, 1)
        norm : torch.Tensor
            batch of OSEM image normalization factors, shape (batch_size)
            "unnormalized image" = norm[i_batch] * osem[i_batch,...]

        Returns
        -------
        torch.Tensor
            batch of image passed through the model
        """

        num_batch = osem.shape[0]

        output_tensor = torch.zeros_like(osem)

        for i in range(num_batch):
            unscaled_image = norm[i] * osem[i, 0, ...]

            # calculate the PET subset  forward step
            expected_data = multiplicative_corrections[
                i, subset, ...] * self._pet_fwd_subset(
                    unscaled_image, self._proj, subset) + contamination[i,
                                                                        subset,
                                                                        ...]

            # Poisson logL update
            unscaled_update = self._pet_adjoint_subset(
                multiplicative_corrections[i, subset, ...] *
                (1 - data[i, subset, ...] / expected_data), self._proj,
                subset) * (1. / adjoint_ones[i, subset, ...])
            #subset) * (unscaled_image / adjoint_ones[i, subset, ...])

            output_tensor[i, 0, ...] = unscaled_update / norm[i]

        return output_tensor

    def forward(self, osem: torch.Tensor, data: torch.Tensor,
                multiplicative_corrections: torch.Tensor,
                contamination: torch.Tensor, adjoint_ones: torch.Tensor,
                norm: torch.Tensor) -> torch.Tensor:
        """forward pass of model

        Parameters
        ----------
        osem : torch.Tensor
            batch of normalized OSEM images, shape (batch_size,0,n0,n1,1)
        data : torch.Tensor
            batch of (subset chunked) emission sinograms, shape (batch_size, num_subsets, num_subset_lors, num_tof_bins)
        multiplicative_corrections : torch.Tensor
            batch of (subset chunked) mult. correction sinograms, shape (batch_size, num_subsets, num_subset_lors, 1)
        contamination : torch.Tensor
            batch of (subset chunked) contamination sinograms, shape (batch_size, num_subsets, num_subset_lors, num_tof_bins)
        adjoint_ones : torch.Tensor
            batch of adjoint ones (subset sensitivity image), shape (batch_size, num_subsets, n0, n1, 1)
        norm : torch.Tensor
            batch of OSEM image normalization factors, shape (batch_size)
            "unnormalized image" = norm[i_batch] * osem[i_batch,...]

        Returns
        -------
        torch.Tensor
            batch of image passed through the model
        """
        x = osem
        for subset in range(self._proj.subsetter.num_subsets):
            x1 = self._data_fidelity_subset_update(x, data,
                                                   multiplicative_corrections,
                                                   contamination, adjoint_ones,
                                                   norm, subset)
            x2 = self._neural_net(x)
            x = torch.nn.ReLU()(x - x1 + self._neural_net_weight * x2)

        return x


if __name__ == '__main__':
    dtype = torch.float32
    device = torch.device("cuda:0")  # Uncomment this to run on GPU
    training_data_dir: str = '../data/OSEM_2D_5.00E+01'
    batch_size: int = 8
    learning_rate: float = 1e-3

    #---------------------------------------------------------------------------
    #--- setup the data loaders ------------------------------------------------
    #---------------------------------------------------------------------------

    ds = OSEM2DDataSet(basedir=training_data_dir)
    training_data_loader = torch.utils.data.DataLoader(ds,
                                                       batch_size=batch_size,
                                                       shuffle=True,
                                                       num_workers=5)

    #---------------------------------------------------------------------------
    #--- load the projector ----------------------------------------------------
    #---------------------------------------------------------------------------

    # the projector (without multiplicative corrections) should be the same for
    # all data sets, so we only restore the one from the first data set
    with open(ds.dir_list[0] / 'projector.pkl', 'rb') as f:
        projector = dill.load(f)

    #---------------------------------------------------------------------------
    #--- define a simple conv net ----------------------------------------------
    #---------------------------------------------------------------------------
    kernel_size = (3, 3, 1)
    conv1 = torch.nn.Conv3d(1,
                            10,
                            kernel_size,
                            padding='same',
                            device=device,
                            dtype=torch.float32)
    conv2 = torch.nn.Conv3d(10,
                            10,
                            kernel_size,
                            padding='same',
                            device=device,
                            dtype=torch.float32)
    conv3 = torch.nn.Conv3d(10,
                            1,
                            kernel_size,
                            padding='same',
                            device=device,
                            dtype=torch.float32)

    conv_net = torch.nn.Sequential(
        collections.OrderedDict([('conv1', conv1), ('conv2', conv2),
                                 ('conv3', conv3),
                                 ('relu3', torch.nn.ReLU())]))

    model = MyModel(projector, conv_net)

    # setup the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    #---------------------------------------------------------------------------
    #--- training loop ---------------------------------------------------------
    #---------------------------------------------------------------------------

    for epoch in range(5):
        for i, (osem, data, multiplicative_corrections, contamination,
                adjoint_ones, norm, image) in enumerate(training_data_loader):

            # send mini-batch of data to cuda device
            # normalized OSEM recon - shape (batch_size, 1, n0, n1, 1)
            osem = osem.to(device)
            # emission sinograms - shape (batch_size, num_subsets, num_subset_lors, num_tof_bins)
            data = data.to(device)
            # mult. correction sinograms - shape (batch_size, num_subsets, num_subset_lors, 1)
            multiplicative_corrections = multiplicative_corrections.to(device)
            # contamination sinograms - shape (batch_size, num_subsets, num_subset_lors, num_tof_bins)
            contamination = contamination.to(device)
            # adjoint ones (sens. images) - shape (batch_size, num_subsets, n0, n1, 1)
            adjoint_ones = adjoint_ones.to(device)
            # normalization factor used to normalize OSEM image - shape (batch_size,)
            norm = norm.to(device)
            # normalized ground truth images (batch_size, 1, n0, n1, 1)
            image = image.to(device)

            x_fwd = model.forward(osem, data, multiplicative_corrections,
                                  contamination, adjoint_ones, norm)

            loss = ((x_fwd - image)**2).mean()
            print(
                f'{epoch:03} / {(i+1):03} / {(ds.__len__() // batch_size):03} loss: {loss:.2E}'
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()