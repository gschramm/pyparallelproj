import torch
import collections


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