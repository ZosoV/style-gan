import torch
import torch.nn as nn

def generate_samples(num_samples, G, device):
    """Function to generate N samples in p-space.

    Args:
        num_samples ([int]): number of samples to be generated
        G           ([torch.Module]): pretrained generator module
        device      ([string]): string with the objective device

    Returns:
        [tensor]: tensor with N samples in p-space. Shape: [num_samples,512] 
    """
    # 1. Sampling 10e6 in Z space # what is the interval of z?
    X = torch.randn((int(num_samples), 512), device=device)
    print("z: ", X.size())

    # 2. Mappint z-space to w-space
    G.mapping.num_ws = 1   # setting to the space w not w+ that was defined by default
    X = G.mapping(X, None) # TODO: check what happend with truncation?

    # 2. Mapping w-space samples to p-space
    X = nn.LeakyReLU(negative_slope=5.0)(X)

    return X

def apply_PCA(samples, q = 512):
    """Perform the PCA of samples X, and return relevant tensors for the embedding algorithm

    Args:
        samples ([tensor]): tensor with n samples in p-space. Shape: [n,512]
        q (int, optional):  number of pricipal components to return. Defaults to 512.

    Returns:
        V [tensor]: the V columns represent the principal directions. Shape [512,512]
        E [tensor]: contains the eigenvalues. Shape [512]
        mean [tensor]: the mean vector of all the samples. Shape [512]
        S [tensor]: contains the singular values: Shape [512]
    """
    n_samples, n_features = samples.size()

    # get the 20 principal components
    U,S,V = torch.pca_lowrank(samples, q=q, center=True, niter=15)

    mean_vector = torch.mean(samples, dim=0)
    print("mean vector: ", mean_vector.size())

    eigen_values = (S**2)/(n_samples - 1)
    print("eigenvalues: ", eigen_values.size())

    print("eigenvectors: ", V.size())
    return V, eigen_values, mean_vector, S

class mapping_P_N(nn.Module):
    """Module to map the current extended space w+ to the extended space p_N+ 
    """
    def __init__(self, C, E, mean):
        """Initial tensor needed

        Args:
            C ([tensor]): Principal Directions from PCA or SVD decomposition. Shape: [512]
            E ([tensor]): Eigenvalues or Singular Values from PCA decomposition. Shape: [512] 
            mean ([tensor]): mean vector. Shape: [512]
        """
        super(mapping_P_N, self).__init__()
        self.C = C
        self.leakyRELU = nn.LeakyReLU(negative_slope=5.0)
        self.E = E
        self.mean = mean.unsqueeze(0).unsqueeze(0)

    def forward(self,input):
        """ Perform the mapping from w+ space to p_N space
        Args:
            input ([tensor]): tensor in w+ space. Shape: [18,512]

        Returns:
            [tensor]: tensor in p_N space. Shape: [18,512]
        """

        # transform to P space
        result = self.leakyRELU(input)

        # rest the mean
        result = (result - self.mean).squeeze(0).T

        # transform to PN space
        result =  torch.matmul(torch.matmul( torch.diag(1/self.E), self.C.T), result)

        return result.T
