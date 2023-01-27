"""minimal example that shows how to setup a cascaded model with layers combining
   (cupy) projections and 3D convolutions
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import dill

import pyparallelproj.petprojectors as petprojectors

from layers import LinearSubsetForwardLayer, LinearSubsetAdjointLayer
from datasets import OSEM2DDataSet
from models import sequential_conv_model


class PETVarNet(torch.nn.Module):
    """dummy cascaded model that includes layers combining projections and convolutions"""

    def __init__(
        self,
        proj: petprojectors.PETJosephProjector,
        neural_net: torch.nn.Module,
        num_blocks: int = 4,
    ) -> None:

        super().__init__()
        self._proj = proj
        self._neural_net = neural_net

        self._num_blocks = num_blocks
        self._subsets_to_use = np.linspace(0,
                                           self._proj.subsetter.num_subsets,
                                           self._num_blocks,
                                           endpoint=False).astype(int)

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
                subset) * (unscaled_image / adjoint_ones[i, subset, ...])

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

        for subset in self._subsets_to_use:
            x_data = self._data_fidelity_subset_update(
                x, data, multiplicative_corrections, contamination,
                adjoint_ones, norm, subset)
            x_net = self._neural_net(x)
            x = torch.nn.ReLU()(osem - x_data +
                                self._neural_net_weight * x_net)

        return x


if __name__ == '__main__':
    dtype = torch.float32
    device = torch.device("cuda:0")
    training_data_dir: str = '../data/training/OSEM_2D_5.00E+01'
    batch_size: int = 16
    learning_rate: float = 1e-3
    num_epochs: int = 100
    loss_fct = torch.nn.L1Loss()
    num_blocks: int = 4
    num_layers: int = 6
    num_features: int = 10

    #---------------------------------------------------------------------------
    #--- setup the data loaders ------------------------------------------------
    #---------------------------------------------------------------------------

    training_data_set = OSEM2DDataSet(training_data_dir)
    training_data_loader = torch.utils.data.DataLoader(training_data_set,
                                                       batch_size=batch_size,
                                                       drop_last=True,
                                                       shuffle=True,
                                                       num_workers=5)

    #---------------------------------------------------------------------------
    #--- load the projector ----------------------------------------------------
    #---------------------------------------------------------------------------

    # the projector (without multiplicative corrections) should be the same for
    # all data sets, so we only restore the one from the first data set
    with open(training_data_set.dir_list[0] / 'projector.pkl', 'rb') as f:
        projector = dill.load(f)

    #---------------------------------------------------------------------------
    #--- define a simple conv net that maps an image onto an image -------------
    #---------------------------------------------------------------------------
    conv_net = sequential_conv_model(device=torch.device("cuda:0"),
                                     kernel_size=(3, 3, 1),
                                     num_layers=num_layers,
                                     num_features=num_features,
                                     dtype=torch.float32)

    model = PETVarNet(projector, conv_net, num_blocks=num_blocks)

    # setup the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    #---------------------------------------------------------------------------
    #--- training loop ---------------------------------------------------------
    #---------------------------------------------------------------------------

    loss_list = []

    for epoch in range(num_epochs):
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

            loss = loss_fct(x_fwd, image)
            loss_list.append(float(loss.cpu().detach().numpy()))

            print(
                f'{epoch:03} / {(i+1):03} / {(ds.__len__() // batch_size):03} loss: {loss:.2E}'
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # save some visualizations at the end of every epoch
        y = conv_net(osem)
        for ib in range(batch_size):
            fig, ax = plt.subplots(2, 2, figsize=(9, 9))
            ax[0, 0].imshow(osem[ib, 0, ...].cpu().detach().numpy().squeeze(),
                            vmin=0,
                            vmax=1.2)
            ax[0, 1].imshow(x_fwd[ib, 0, ...].cpu().detach().numpy().squeeze(),
                            vmin=0,
                            vmax=1.2)
            ax[1, 0].imshow(image[ib, 0, ...].cpu().detach().numpy().squeeze(),
                            vmin=0,
                            vmax=1.2)
            ax[1, 1].imshow(y[ib, 0, ...].cpu().detach().numpy().squeeze())

            for axx in ax.ravel():
                axx.set_axis_off()
            fig.tight_layout()
            fig.savefig(f'sample_{ib:02}.png')
            plt.close()

            if len(loss_list) > 505:
                loss_array = np.array(loss_list)
                fig2, ax2 = plt.subplots(2, 2, figsize=(8, 8))
                ax2[0, 0].loglog(
                    np.convolve(loss_array, np.ones(5) / 5, mode='valid'))
                ax2[0, 1].loglog(
                    np.convolve(loss_array, np.ones(25) / 25, mode='valid'))
                ax2[1, 0].loglog(
                    np.convolve(loss_array, np.ones(100) / 100, mode='valid'))
                ax2[1, 1].loglog(
                    np.convolve(loss_array, np.ones(500) / 500, mode='valid'))
                fig2.tight_layout()
                fig2.savefig(f'loss.png')
                plt.close()
