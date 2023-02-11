import numpy as np
import torch
import collections
import pyparallelproj.petprojectors as petprojectors
from layers import LinearSubsetForwardLayer, LinearSubsetAdjointLayer


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
            x = torch.nn.ReLU()(x - x_data + self._neural_net_weight * x_net)

        return x


#_--------------------------------------------------------------------------------


def sequential_conv_model(device=torch.device("cuda:0"),
                          kernel_size=(3, 3, 1),
                          num_layers=6,
                          num_features=10,
                          dtype=torch.float32) -> torch.nn.Sequential:
    """simple sequential model consisting of 3D conv layers and PReLUs

    Parameters
    ----------
    device : optional
        by default torch.device("cuda:0")
    kernel_size : tuple, optional
        kernel size of conv layers, by default (3, 3, 1)
    num_layers : int, optional
        number of conv layers, by default 6
    num_features : int, optional
        number of features, by default 10
    dtype : optional
        data type for conv layers, by default torch.float32

    Returns
    -------
    Sequential model
    """

    conv_net = collections.OrderedDict()

    conv_net['conv_1'] = torch.nn.Conv3d(1,
                                         num_features,
                                         kernel_size,
                                         padding='same',
                                         device=device,
                                         dtype=dtype)
    conv_net['prelu_1'] = torch.nn.PReLU(device=device)

    for i in range(num_layers - 2):
        conv_net[f'conv_{i+2}'] = torch.nn.Conv3d(num_features,
                                                  num_features,
                                                  kernel_size,
                                                  padding='same',
                                                  device=device,
                                                  dtype=dtype)
        conv_net[f'prelu_{i+2}'] = torch.nn.PReLU(device=device)

    conv_net[f'conv_{num_layers}'] = torch.nn.Conv3d(num_features,
                                                     1,
                                                     kernel_size,
                                                     padding='same',
                                                     device=device,
                                                     dtype=dtype)
    conv_net[f'prelu_{num_layers}'] = torch.nn.PReLU(device=device)

    conv_net = torch.nn.Sequential(conv_net)

    return conv_net


class Unet3D(torch.nn.Module):

    def __init__(self,
                 device,
                 num_features: int = 8,
                 kernel_size: tuple[int, int, int] = (3, 3, 1),
                 dtype=torch.float32) -> None:

        super().__init__()

        self._device = device
        self._num_features = num_features
        self._kernel_size = kernel_size
        self._dtype = dtype

        self._encoder_block_1 = self._conv_block(1, num_features, num_features)
        self._encoder_block_2 = self._conv_block(num_features,
                                                 2 * num_features,
                                                 2 * num_features)
        self._encoder_block_3 = self._conv_block(2 * num_features,
                                                 4 * num_features,
                                                 4 * num_features)
        self._encoder_block_4 = self._conv_block(4 * num_features,
                                                 8 * num_features,
                                                 8 * num_features)

        self._pool = torch.nn.MaxPool3d((2, 2, 1))

        self._upsample_4 = torch.nn.ConvTranspose3d(8 * num_features,
                                                    4 * num_features,
                                                    kernel_size=(2, 2, 1),
                                                    stride=2,
                                                    device=device)

        self._upsample_3 = torch.nn.ConvTranspose3d(4 * num_features,
                                                    2 * num_features,
                                                    kernel_size=(2, 2, 1),
                                                    stride=2,
                                                    device=device)

        self._upsample_2 = torch.nn.ConvTranspose3d(2 * num_features,
                                                    num_features,
                                                    kernel_size=(2, 2, 1),
                                                    stride=2,
                                                    device=device)

        self._decoder_block_3 = self._conv_block(8 * num_features,
                                                 4 * num_features,
                                                 4 * num_features)

        self._decoder_block_2 = self._conv_block(4 * num_features,
                                                 2 * num_features,
                                                 2 * num_features)

        self._decoder_block_1 = self._conv_block(2 * num_features,
                                                 num_features, num_features)

        self._final_conv = torch.nn.Conv3d(num_features,
                                           1, (1, 1, 1),
                                           padding='same',
                                           device=self._device,
                                           dtype=self._dtype)

    def _conv_block(self,
                    num_features_in,
                    num_features_mid,
                    num_features_out,
                    activation=torch.nn.ReLU()):
        conv_block = collections.OrderedDict()

        conv_block['conv_1'] = torch.nn.Conv3d(num_features_in,
                                               num_features_mid,
                                               self._kernel_size,
                                               padding='same',
                                               device=self._device,
                                               dtype=self._dtype)
        conv_block['activation_1'] = activation

        conv_block['conv_2'] = torch.nn.Conv3d(num_features_mid,
                                               num_features_out,
                                               self._kernel_size,
                                               padding='same',
                                               device=self._device,
                                               dtype=self._dtype)
        conv_block['activation_2'] = activation

        conv_block = torch.nn.Sequential(conv_block)

        return conv_block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self._encoder_block_1(x)
        x2 = self._encoder_block_2(self._pool(x1))
        x3 = self._encoder_block_3(self._pool(x2))

        # the last encoder block is the bottleneck
        x4 = self._encoder_block_4(self._pool(x3))

        x3up = self._decoder_block_3(
            torch.cat([x3, self._upsample_4(x4)], dim=1))

        x2up = self._decoder_block_2(
            torch.cat([x2, self._upsample_3(x3up)], dim=1))

        x1up = self._decoder_block_1(
            torch.cat([x1, self._upsample_2(x2up)], dim=1))

        xout = self._final_conv(x1up)

        return xout


if __name__ == '__main__':
    device = torch.device("cuda:0")
    dtype = torch.float32
    x = torch.rand(4, 1, 128, 128, 1, dtype=dtype).to(device)

    model = Unet3D(device, dtype=dtype)