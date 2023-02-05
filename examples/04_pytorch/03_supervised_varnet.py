"""minimal example that shows how to setup a cascaded model with layers combining
   (cupy) projections and 3D convolutions
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import torch
import dill

import pyparallelproj.petprojectors as petprojectors

from pathlib import Path
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
            x = torch.nn.ReLU()(x - x_data + self._neural_net_weight * x_net)

        return x


#---------------------------------------------------------------------


def training_loop(dataloader, model, loss_fn, optimizer):

    loss_list = []
    model.train()

    for i, (osem, data, multiplicative_corrections, contamination,
            adjoint_ones, norm, image) in enumerate(dataloader):

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

        loss = loss_fn(x_fwd, image)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_list.append(float(loss.cpu().detach().numpy()))

        if i % 30 == 0:
            print(
                f'{(i+1):03} / {(dataloader.dataset.__len__() // batch_size):03} loss: {loss_list[-1]:.2E}'
            )

    return loss_list


#------------------------------------------------------------------------------------------------------


def validation_loop(dataloader, model, loss_fn, save_dir):

    num_batches = len(dataloader)
    val_loss = 0
    model.eval()

    with torch.no_grad():
        for i, (osem, data, multiplicative_corrections, contamination,
                adjoint_ones, norm, image) in enumerate(dataloader):

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

            val_loss += loss_fn(x_fwd, image)

    val_loss /= num_batches

    print(f'validation loss: {val_loss:.2E}')

    show_validation_batch(osem, x_fwd, image, model, save_dir)

    return val_loss


def show_validation_batch(input, output, gt, mdl, save_dir):

    conv_output = mdl._neural_net(input).cpu().detach().numpy().squeeze()
    conv_output2 = mdl._neural_net(output).cpu().detach().numpy().squeeze()
    conv_output3 = mdl._neural_net(gt).cpu().detach().numpy().squeeze()

    for ib in range(input.shape[0]):
        fig, ax = plt.subplots(2, 6, figsize=(18, 6))
        ground_truth = gt[ib, 0, ...].cpu().detach().numpy().squeeze()
        ax[0, 0].imshow(ground_truth, vmin=0, vmax=1.2)
        ax[0, 1].imshow(input[ib, 0, ...].cpu().detach().numpy().squeeze(),
                        vmin=0,
                        vmax=1.2)
        ax[0, 2].imshow(gaussian_filter(
            input[ib, 0, ...].cpu().detach().numpy().squeeze(), 1.),
                        vmin=0,
                        vmax=1.2)
        ax[0, 3].imshow(gaussian_filter(
            input[ib, 0, ...].cpu().detach().numpy().squeeze(), 1.7),
                        vmin=0,
                        vmax=1.2)
        ax[0, 4].imshow(output[ib, 0, ...].cpu().detach().numpy().squeeze(),
                        vmin=0,
                        vmax=1.2)

        dmax = 0.3
        ax[1, 1].imshow(input[ib, 0, ...].cpu().detach().numpy().squeeze() -
                        ground_truth,
                        vmin=-dmax,
                        vmax=dmax,
                        cmap=plt.cm.seismic)
        ax[1, 2].imshow(gaussian_filter(
            input[ib, 0, ...].cpu().detach().numpy().squeeze(), 1.) -
                        ground_truth,
                        vmin=-dmax,
                        vmax=dmax,
                        cmap=plt.cm.seismic)
        ax[1, 3].imshow(gaussian_filter(
            input[ib, 0, ...].cpu().detach().numpy().squeeze(), 1.7) -
                        ground_truth,
                        vmin=-dmax,
                        vmax=dmax,
                        cmap=plt.cm.seismic)
        ax[1, 4].imshow(output[ib, 0, ...].cpu().detach().numpy().squeeze() -
                        ground_truth,
                        vmin=-dmax,
                        vmax=dmax,
                        cmap=plt.cm.seismic)

        vmax = 0.5 * np.abs(conv_output[ib, ...]).max()
        ax[0, 5].imshow(conv_output[ib, ...],
                        vmin=-vmax,
                        vmax=vmax,
                        cmap=plt.cm.seismic)
        ax[1, 5].imshow(conv_output2[ib, ...],
                        vmin=-vmax,
                        vmax=vmax,
                        cmap=plt.cm.seismic)
        ax[1, 0].imshow(conv_output3[ib, ...],
                        vmin=-vmax,
                        vmax=vmax,
                        cmap=plt.cm.seismic)

        for axx in ax.ravel():
            axx.set_axis_off()
        fig.tight_layout()
        fig.savefig(save_dir / f'sample_{ib:02}.png')
        plt.close()


def plot_training_loss(loss_list: list[float], save_dir: Path):
    loss_array = np.array(loss_list)
    fig2, ax2 = plt.subplots(2, 2, figsize=(8, 8))
    if loss_array.size > 5:
        sm_ker = np.ones(5) / 5
        ax2[0, 0].loglog(np.convolve(loss_array, sm_ker, mode='valid'))
    if loss_array.size > 25:
        sm_ker = np.ones(25) / 25
        ax2[0, 1].loglog(np.convolve(loss_array, sm_ker, mode='valid'))
    if loss_array.size > 100:
        sm_ker = np.ones(100) / 100
        ax2[1, 0].loglog(np.convolve(loss_array, sm_ker, mode='valid'))
    if loss_array.size > 500:
        sm_ker = np.ones(500) / 500
        ax2[1, 1].loglog(np.convolve(loss_array, sm_ker, mode='valid'))
    fig2.tight_layout()
    fig2.savefig(save_dir / f'training_loss.png')
    plt.close()


def plot_validation_loss(loss: np.ndarray, save_dir: Path):
    inds = np.where(loss > 0)
    fig3, ax3 = plt.subplots(1, 3, figsize=(9, 3))
    ax3[0].plot(loss[inds])
    if loss[inds].size > 3:
        sm_ker = np.ones(3) / 3
        ax3[1].plot(np.convolve(loss[inds], sm_ker, mode='valid'))
    if loss[inds].size > 5:
        sm_ker = np.ones(5) / 5
        ax3[2].plot(np.convolve(loss[inds], sm_ker, mode='valid'))
    fig3.tight_layout()
    fig3.savefig(save_dir / f'validation_loss.png')
    plt.close()


#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--num_blocks', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--num_features', type=int, default=10)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--input_seed', type=int, default=1)
    parser.add_argument('--target_image_name', type=str, default='image.npz')
    parser.add_argument('--training_data_dir',
                        type=str,
                        default='../data/training/OSEM_2D_5.00E+01')
    parser.add_argument('--validation_data_dir',
                        type=str,
                        default='../data/validation/OSEM_2D_5.00E+01')
    parser.add_argument('--loss',
                        type=str,
                        default='L1',
                        choices=['L1', 'MSE'])
    args = parser.parse_args()

    if args.ckpt is None:
        checkpoint = None
    else:
        # in case we load a checkpoint, we have to make sure
        # that num_layers, num_features, num_block are consistent
        checkpoint = torch.load(args.ckpt)
        with open(Path(args.ckpt).parent / 'config.json', 'r') as f:
            config = json.load(f)
        args.num_blocks = config['num_blocks']
        args.num_layers = config['num_layers']
        args.num_features = config['num_features']

    num_epochs: int = args.num_epochs
    batch_size: int = args.batch_size
    learning_rate: float = args.learning_rate
    num_blocks: int = args.num_blocks
    num_layers: int = args.num_layers
    num_features: int = args.num_features
    input_seed: int = args.input_seed
    target_image_name: str = args.target_image_name
    loss: str = args.loss

    training_data_dir: str = args.training_data_dir
    validation_data_dir: str = args.validation_data_dir

    dtype = torch.float32
    device = torch.device("cuda:0")

    if loss == 'L1':
        loss_fct = torch.nn.L1Loss()
    elif loss == 'MSE':
        loss_fct = torch.nn.MSELoss()
    else:
        raise ValueError

    i_out = 0
    output_dir = Path(f'run/{i_out:04}')

    while output_dir.exists():
        i_out += 1
        output_dir = Path(f'run/{i_out:04}')

    output_dir.mkdir(exist_ok=True, parents=True)

    # save input configuration
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f)

    #---------------------------------------------------------------------------
    #--- setup the data loaders ------------------------------------------------
    #---------------------------------------------------------------------------

    training_data_set = OSEM2DDataSet(training_data_dir)
    training_data_set.seed = input_seed
    training_data_set.target_image_name = target_image_name

    training_data_loader = torch.utils.data.DataLoader(
        training_data_set,
        batch_size=batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=5,
        pin_memory=True,
        pin_memory_device='cuda:0')

    validation_data_set = OSEM2DDataSet(validation_data_dir)
    validation_data_set.seed = input_seed
    validation_data_set.target_image_name = target_image_name
    validation_data_loader = torch.utils.data.DataLoader(
        validation_data_set,
        batch_size=batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=5,
        pin_memory=True,
        pin_memory_device='cuda:0')

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
    conv_net = sequential_conv_model(device=device,
                                     kernel_size=(3, 3, 1),
                                     num_layers=num_layers,
                                     num_features=num_features,
                                     dtype=torch.float32)

    model = PETVarNet(projector, conv_net, num_blocks=num_blocks)

    # setup the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if checkpoint is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        optimizer.param_groups[0]['lr'] = learning_rate
        start_epoch = checkpoint['epoch']
        model.train()
    else:
        start_epoch = 0

    #---------------------------------------------------------------------------
    #--- training loop ---------------------------------------------------------
    #---------------------------------------------------------------------------

    training_loss = []
    validation_loss = np.zeros(num_epochs)

    for epoch in np.arange(num_epochs):
        print(f'epoch {(epoch+1):04} / {num_epochs:04}')
        training_loss += training_loop(training_data_loader, model, loss_fct,
                                       optimizer)
        validation_loss[epoch] = validation_loop(validation_data_loader, model,
                                                 loss_fct, output_dir)

        # save the last checkpoint
        torch.save(
            {
                'epoch': epoch + start_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': training_loss[-1],
            }, output_dir / 'last_model.cpt')

        # if the validation loss in minimal so far, save to a separate file
        if validation_loss[epoch] == validation_loss[
                validation_loss > 0].min():
            torch.save(
                {
                    'epoch': epoch + start_epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': training_loss[-1],
                }, output_dir / 'model_best_val_loss.cpt')

        # plot the loss functions to files
        plot_training_loss(training_loss, output_dir)
        plot_validation_loss(validation_loss, output_dir)