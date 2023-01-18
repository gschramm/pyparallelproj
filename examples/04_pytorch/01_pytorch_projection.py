"""minimal example that shows how to setup and train a torch model that includes combines the output
   of a NN and a cupy-based data fidelity term"""
import torch
import cupy as cp


class MiniCupyNetwork(torch.nn.Module):

    def __init__(self) -> None:
        self._device = device
        super(MiniCupyNetwork, self).__init__()
        # setup a simple mode - in real life probably more complicated
        self.simple_conv_stack = torch.nn.Sequential(
            torch.nn.Conv3d(1, 10, (3, 3, 3), padding='same'),
            torch.nn.ReLU(),
            torch.nn.Conv3d(10, 1, (3, 3, 3), padding='same'),
            torch.nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # feed image batch through the neural network
        network_output = self.simple_conv_stack(x)

        # calculate the data fidelity gradient - here only constant array
        data_fidelity_grad = cp.zeros(network_output.shape, dtype=cp.float32)
        ## in real life we would do sth like
        cupy_x = cp.ascontiguousarray(cp.from_dlpack(x.detach()))
        for i_batch in range(cupy_x.shape[0]):
            # dummy gradient
            data_fidelity_grad[i_batch, 0, ...] = 0.5
        #   in real life we would do sth like
        #   cupy_image = cupy_x[i_batch,0,...]
        #   x_fwd = self.projector.forward()
        #   data_fidelity_grad[i, 0, ...] = self.projector.adjoint(1 - data/x_fwd)

        # convert data fidelity to torch tensor and add to network output
        torch_y = torch.from_dlpack(data_fidelity_grad) + network_output

        return torch_y


#-------------------------------------------------------------

if __name__ == '__main__':
    torch.manual_seed(0)
    lr = 1e-3
    device = "cuda"
    model = MiniCupyNetwork().to(device)
    print(model)

    # generate a random batch of 2 single channel images with spatial shape (7,7,7)
    x_batch = torch.randn(2, 1, 7, 7, 7, device=device)
    x_batch_fwd = model.forward(x_batch)
    y_batch = torch.rand(x_batch_fwd.shape, device=device)

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for i_update in range(1000):
        pred = model(x_batch)
        loss = loss_fn(pred, y_batch)

        if (i_update) % 10 == 0:
            print(f"Epoch {(i_update):04}, loss {loss:.3e}")

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("Done!")