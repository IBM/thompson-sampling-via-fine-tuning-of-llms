import torch
from typing import Tuple
from transformers import AutoTokenizer
from src.utils import misc
from src import lite

rff_random_normals = None # cache that ensures a consistent feature map is used
def _get_rff_random_normals(feature_dim:int, n_random_fourier_features:int, device:torch.device, dtype=torch.dtype) -> torch.Tensor:
    """
    Returns a matrix of sampled standard normals of shape (feature_dim, n_random_fourier_features//2) on the prescribed device unless the matrix 
    is already cached. The matrix of normals is used to compute random Fourier features (RFFs) according to the function `random_fourier_features`.

    Args:
        feature_dim (torch.Tensor): The dimensionality of the original feature space.
        n_random_fourier_features (torch.Tensor): The dimensionality of the transformed feature space, aka the number of random fourier features.
        device (torch.device): The device on which the random normals should be placed.
        dtype (torch.dtype): The data type to be used for the random normals.

    Returns:
        out (torch.Tensor): A matrix of sampled standard normals of shape (feature_dim, n_random_fourier_features//2)
    """
    global rff_random_normals
    if rff_random_normals is None:
        rff_random_normals = torch.randn(feature_dim, n_random_fourier_features//2, device=device, dtype=dtype)
    else:
        assert rff_random_normals.shape == (feature_dim, n_random_fourier_features//2), f"rff_random_normals.shape = {rff_random_normals.shape}"
    return rff_random_normals

def random_fourier_features(features:torch.Tensor, rff_parameters:Tuple[float, int]) -> torch.Tensor:
    """
    Transforms `features` into `random_fourier_features` that approximate the RBF kernel on the feature space.

    Args:
        features (torch.Tensor): The features on which the rbf kernel should be applied using the approximation with random fourier features.
        rff_parameters (Tuple[float, int]): The RBF length scale and the number of random fourier features.
    Returns:
        out (torch.Tensor): A tensor containing batched random fourier features for the `features` tensor.
    """
    length_scale, n_random_fourier_features,  = rff_parameters
    rff_random_normals = _get_rff_random_normals(features.shape[1], n_random_fourier_features, features.device, features.dtype)
    return misc.random_fourier_features(features, n_random_fourier_features, length_scale, rff_random_normals)

def sequence_mean(features:torch.Tensor, X:torch.Tensor, tokenizer:AutoTokenizer) -> torch.Tensor:
    """
    Extracts the sequence mean feature from the `features` tensor along the text-generation dimension. 
    The sequence mean is computed by setting all token embeddings after the first `<eos>` to zero and then 
    computing the mean along the text-generation dimension after token-level normalization. This corresponds to the average of the generated text. 

    Args:
        features (torch.Tensor): The features on which the sequence mean should be computed.
        X (torch.Tensor): The input tokens over whose features we aggregate.
        tokenizer (AutoTokenizer): The tokenizer that converts between text and tokens. Used here to identify `<eos>` tokens.

    Returns:
        out (torch.Tensor): A tensor containing batched sequence means for the `features` tensor.
    """
    features = torch.nn.functional.normalize(features, p=2, dim=-1)
    mask = torch.cumsum(X == tokenizer.eos_token_id, dim=1) > 1 # masks all but the first <eos> token
    features[mask, :] = 0
    return torch.mean(features, dim=1)

def sequence_latest(features:torch.Tensor, X:torch.Tensor, tokenizer:AutoTokenizer) -> torch.Tensor:
    """
    Extracts the last sequence feature from the `features` tensor along the text-generation dimension. 
    The last feature position is identified as the first occurance of the token `<eos>`.

    Args:
        features (torch.Tensor): The features whose last feature entry should be extracted along the text-generation dimension.
        X (torch.Tensor): The input tokens over whose features we aggregate.
        tokenizer (AutoTokenizer): The tokenizer that converts between text and tokens. Used here to identify `<eos>` tokens.

    Returns:
        out (torch.Tensor): A tensor containing batched latest entries of the `features` tensor.
    """
    latest_token_indices = torch.argmax((X == tokenizer.eos_token_id).to(torch.int), dim=1) # extracts position of first <eos> token
    return features[torch.arange(X.shape[0]), latest_token_indices, :]

def add_bias(features:torch.Tensor, bias:float=1.0) -> torch.Tensor:
    """
    Takes a feature map and adds a constant column of value `bias` to allow a non-zero mean (see constant kernel).

    Args:
        features (torch.Tensor): The feature map to which the bias should be added.
        bias (float, optional): The bias term, defaults to 1.0.

    Returns:
        out (torch.Tensor): The features augmented with a constant entry of value `bias`.
    """
    return torch.cat((features, torch.full((features.shape[0], 1), bias, dtype=features.dtype, device=features.device)), dim=1) # adds bias

def normalize(features:torch.Tensor) -> torch.Tensor:
    """
    Returns the column-wise L2 normalized features.

    Args:
        features (torch.Tensor): The features to be L2 normalized.

    Returns:
        out (torch.Tensor): The L2 normalized features.
    """
    return torch.nn.functional.normalize(features, p=2, dim=1)


def add_prior_prob(features:torch.Tensor, prior_log_gen_prob:torch.Tensor) -> torch.Tensor:
    """
    Takes a feature map and adds a column of value v^{-1}(pi_x^{theta_0}) to incorporate prior probability of maximality into the kernel.

    Args:
        features (torch.Tensor): The feature map to which the prior information should be added
        prior_log_gen_prob (torch.Tensor): The prior log generation probability

    Returns:
        out (torch.Tensor): The features augmented with v^{-1}(pi_x^{theta_0})
    """
    return torch.cat(features, lite.inv_vapor_exp(prior_log_gen_prob).unsqueeze(1), dim=1)