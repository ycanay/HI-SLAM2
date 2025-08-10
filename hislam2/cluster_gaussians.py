import torch
import torch_fpsample
from sklearn.cluster import KMeans
import numpy as np


def fps_indices(gaussians: torch.Tensor, num_clusters: int) -> torch.Tensor:
    """
    Performs farthest point sampling on the given gaussians tensor.
    args:
        gaussians (torch.Tensor): A tensor of shape (N, 3) where N
        num_clusters (int): The number of clusters to sample.
    returns:
        torch.Tensor: A tensor of shape (num_clusters, D) containing the sampled gaussian indices.
    """
    assert gaussians.dim() == 2, "gaussians must be a 2D tensor."
    assert gaussians.shape[0] >= num_clusters, "num_clusters must be less than or equal to the number of gaussians."
    assert gaussians.shape[1] == 3, "gaussians must have at least one dimension."
    points, indices = torch_fpsample.sample(gaussians, num_clusters)
    return indices

# TODO: Later implement own version


def incremental_clustering(gaussians: torch.Tensor, ins_feats: torch.Tensor, num_clusters: int) -> torch.Tensor:
    """
    Incrementally clusters the gaussians tensor by sampling a subset of indices.
    Starting with the fps indices, itereatively creates clusters according to their distance and instance features.
    args:
        gaussians (torch.Tensor): A tensor of shape (N, 3) where N is the number of gaussians and D is the dimension.
        ins_feates (torch.Tensor): A tensor of shape (N, D) where N is the number of gaussians and D is the dimension of the instance features.
        num_clusters (int): The number of clusters to sample.
    returns:
        torch.Tensor: A tensor of shape (num_clusters, D) containing the sampled gaussian indices.
    """
    assert gaussians.dim() == 2, "gaussians must be a 2D tensor."
    assert ins_feats.dim() == 2, "ins_feates must be a 2D tensor."
    assert gaussians.shape[0] == ins_feats.shape[0], "gaussians and ins_feates must have the same number of rows."
    assert gaussians.shape[1] == 3, "gaussians must have at least one dimension."
    assert ins_feats.shape[1] == 6, "ins_feates must have at least one dimension."

    ins_feat_threshold = 0.05
    fps_ids = fps_indices(gaussians, num_clusters)

    all_indices = list(range(gaussians.shape[0]))


def over_segmentation(gaussians: torch.Tensor, features: torch.Tensor, num_clusters: int) -> torch.Tensor:
    """
    Over-segments the gaussians tensor by sampling a subset of indices.
    args:
        gaussians (torch.Tensor): Positions of gaussians of shape (N, 3) where N is the number of gaussians.
        features (torch.Tensor): Instance features of gaussians of shape (N, 6).
        num_clusters (int): The number of clusters to sample.
    returns:
        torch.Tensor: A tensor of shape (num_clusters, D) containing the sampled gaussian indices.
    """
    assert gaussians.dim() == 2, "gaussians must be a 2D tensor."
    assert gaussians.shape[1] == 3, "gaussians must have at least one dimension."
    assert features.dim() == 2, "features must be a 2D tensor."
    assert features.shape[1] == 6, "features must have at least one dimension."
    assert gaussians.shape[0] >= num_clusters, "num_clusters must be less than or equal to the number of gaussians."

    positional_encodings = gaussians.cpu()
    concat_vector = torch.cat((positional_encodings, features), dim=1)
    concat_vector = concat_vector - concat_vector.mean(dim=0, keepdim=True)
    concat_vector = concat_vector / \
        (concat_vector.std(dim=0, keepdim=True) + 1e-6)
    cluster_centers, cluster_indexes = torch_fpsample.sample(
        concat_vector, num_clusters)
    kmeans = KMeans(n_clusters=num_clusters, init=cluster_centers)
    kmeans.fit(concat_vector.cpu().numpy())
    return torch.Tensor(kmeans.labels_)


def positional_encoding_3d_batch(coords: torch.Tensor, dim_total: int = 3) -> torch.Tensor:
    """
    coords: torch.Tensor of shape (N, 3)
    dim_total: total output dimension (must be divisible by 3)
    Returns: torch.Tensor of shape (N, dim_total)
    """
    assert dim_total % 3 == 0, "dim_total must be divisible by 3"
    dim_per_coord = dim_total // 3
    device = coords.device

    def encode(coord_column: torch.Tensor) -> torch.Tensor:
        N = coord_column.shape[0]
        div_term = torch.exp(
            torch.arange(0, dim_per_coord, 2, device=device).float(
            ) * (-torch.log(torch.tensor(10000.0)) / dim_per_coord)
        )  # (dim_per_coord / 2,)
        # (N, dim_per_coord / 2)
        sin_term = torch.sin(coord_column[:, None] * div_term)
        cos_term = torch.cos(coord_column[:, None] * div_term)
        pe = torch.zeros(N, dim_per_coord, device=device)
        pe[:, 0::2] = sin_term
        pe[:, 1::2] = cos_term
        return pe

    x_pe = encode(coords[:, 0])
    y_pe = encode(coords[:, 1])
    z_pe = encode(coords[:, 2])

    return torch.cat([x_pe, y_pe, z_pe], dim=1)  # (N, 12)
