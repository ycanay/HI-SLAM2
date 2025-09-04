from hislam2.cluster_gaussians import over_segmentation
import torch
from itertools import product
import logging
logger = logging.getLogger(__name__)
logger.setLevel('INFO')
FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO)


def normalize_vector(features):
    """
    Normalize features to the range [0, 1].
    """
    features_copy = features.clone()
    features_std = features.std(dim=0, keepdim=True)
    features_mean = features.mean(dim=0, keepdim=True)
    normalized_features = (features_copy - features_mean) / features_std
    return normalized_features


def create_cluster_dic(labels: torch.Tensor, ins_feats: torch.Tensor) -> list:
    """
    Creates a dictionary of clusters with indices and mean of ins_feats.

    Args:
        labels (torch.Tensor): Cluster labels for each point.
        ins_feats (torch.Tensor): Instance features for each point.

    Returns:
        dict: A dictionary where each key is a cluster label, and each value is a dict with:
              - 'indices': indices of points in that cluster
              - 'mean': mean of ins_feats in that cluster
    """
    cluster_dic = []

    for label in range(torch.unique(labels).shape[0]):
        indices = torch.where(labels == label)[0]
        feats = ins_feats[indices]
        mean_feats = torch.mean(feats, dim=0)
        cluster_dic.append(
            {
                'indices': indices,
                'mean': mean_feats,
            }
        )

    return cluster_dic


def build_adjacency_matrix(points: torch.Tensor, labels: torch.Tensor, clusters: list, resolution: int = 64, similarity_threshold: float = 1.5) -> torch.Tensor:
    """
    Builds adjacency matrix based on voxel proximity and feature similarity.
    Args:
        points (torch.Tensor): 3D coordinates of points of shape (N, 3), float.
        labels (torch.Tensor): Cluster labels for each point of shape (N,), long/int.
        clusters (list): List of clusters with indexes and 'mean' features .
        resolution (int): Resolution of the voxel grid.
        similarity_threshold (float): Threshold for feature similarity.
    Returns:
        torch.Tensor: Binary adjacency matrix of shape (K, K), where K is the number of clusters.
    """

    device = points.device

    min_vals = points.min(dim=0, keepdim=True).values
    max_vals = points.max(dim=0, keepdim=True).values
    points_scaled = (points - min_vals) / (max_vals - min_vals + 1e-9)
    points_scaled = (points_scaled * (resolution - 1)).round().to(torch.int64)

    lin_idx = (
        points_scaled[:, 0] * (resolution**2)
        + points_scaled[:, 1] * resolution
        + points_scaled[:, 2]
    )

    unique_voxels, inverse_indices = torch.unique(lin_idx, return_inverse=True)
    voxel_coords = torch.stack([
        unique_voxels // (resolution**2),
        (unique_voxels % (resolution**2)) // resolution,
        unique_voxels % resolution
    ], dim=1)

    # Build a mapping from voxel coordinate tuple to index for fast lookup
    voxel_coord_map = {tuple(coord.tolist()): idx for idx,
                       coord in enumerate(voxel_coords)}

    voxel_clusters = [[] for _ in range(len(unique_voxels))]
    for i, lbl in zip(inverse_indices.tolist(), labels.tolist()):
        voxel_clusters[i].append(lbl)
    voxel_clusters = [torch.tensor(v, dtype=torch.long, device=device).unique()
                      for v in voxel_clusters]

    K = int(labels.max().item()) + 1
    avg_features = torch.stack([
        clusters[i]['mean'].to(device) if isinstance(clusters[i]['mean'], torch.Tensor)
        else torch.tensor(clusters[i]['mean'], dtype=torch.float32, device=device)
        for i in range(K)
    ], dim=0)  # shape (K, F)

    # Precompute pairwise feature distances
    feature_dists = torch.cdist(
        avg_features, avg_features, p=2)  # shape (K, K)
    similarity_mask = feature_dists < similarity_threshold

    adj_matrix = torch.zeros((K, K), dtype=torch.bool, device=device)
    neighbor_offsets = torch.tensor(
        [o for o in product([-1, 0, 1], repeat=3) if o != (0, 0, 0)],
        dtype=torch.int64,
        device=device
    )

    for idx, cluster_set in enumerate(voxel_clusters):
        # Same voxel connections (vectorized)
        if len(cluster_set) > 1:
            cset = cluster_set.unsqueeze(0)
            cset2 = cluster_set.unsqueeze(1)
            mask = (
                cset != cset2) & similarity_mask[cluster_set][:, cluster_set]
            adj_matrix[cluster_set.unsqueeze(1), cluster_set] |= mask

        voxel_pos = voxel_coords[idx]
        neighbors = voxel_pos + neighbor_offsets
        for npos in neighbors:
            npos_tuple = tuple(npos.tolist())
            neighbor_idx = voxel_coord_map.get(npos_tuple, None)
            if neighbor_idx is not None:
                c1s = cluster_set
                c2s = voxel_clusters[neighbor_idx]
                # Vectorized pairwise similarity for neighbor clusters
                c1s_exp = c1s.unsqueeze(1)
                c2s_exp = c2s.unsqueeze(0)
                mask = (c1s_exp != c2s_exp) & similarity_mask[c1s][:, c2s]
                adj_matrix[c1s_exp, c2s] |= mask

    # Convert to uint8 for compatibility
    return adj_matrix.to(torch.uint8)


def connected_components(adj_matrix: torch.Tensor):
    """
    Computes connected components from a binary adjacency matrix.

    Args:
        adj_matrix (torch.Tensor): Shape (N, N), with 1s indicating edges.

    Returns:
        List[List[int]]: A list of components; each component is a list of node indices.
    """
    num_nodes = adj_matrix.shape[0]
    visited = torch.zeros(num_nodes, dtype=torch.bool)
    components = []

    for i in range(num_nodes):
        if not visited[i]:
            stack = [i]
            component = []

            while stack:
                node = stack.pop()
                if not visited[node]:
                    visited[node] = True
                    component.append(node)
                    neighbors = (adj_matrix[node] == 1).nonzero(
                        as_tuple=False).flatten().tolist()
                    stack.extend(neighbors)

            components.append(component)

    return components


def merge_clusters_by_components(clusters, components, ins_feats):
    """
    Merges clusters based on connected components.

    Args:
        clusters (List[Dict]): Original clusters.
        components (List[List[int]]): Groupings of cluster indices.

    Returns:
        List[Dict[str, torch.Tensor]]: Merged clusters.
    """
    merged_clusters = []

    for comp in components:
        merged_indices = torch.cat(
            [clusters[i]['indices'] for i in comp], dim=0)
        merged_features = ins_feats[merged_indices]
        merged_clusters.append({
            'indices': merged_indices,
            'avg_feature': merged_features.mean(dim=0)
        })

    return merged_clusters


def create_clusters(labels, normalized_ins_feats, gaussian_pos, similarity_threshold=.25, resolution=32, verbose=True):
    if verbose:
        logging.info(
            f"Creating clusters with similarity threshold {similarity_threshold} and resolution {resolution}.")
    clusters = create_cluster_dic(labels, normalized_ins_feats)
    if verbose:
        logging.info('Building adjacency matrix...')
    adj_matrix = build_adjacency_matrix(
        gaussian_pos, labels, clusters, resolution, similarity_threshold)
    components = connected_components(adj_matrix)
    merged_clusters = merge_clusters_by_components(
        clusters, components, normalized_ins_feats)
    if verbose:
        logging.info(f"Found {len(merged_clusters)} merged clusters.")
    new_labels = torch.zeros_like(labels)
    for new_label, cluster in enumerate(merged_clusters):
        indices = cluster['indices']
        new_labels[indices] = new_label
    return new_labels, merged_clusters


def create_clusters_iterative(ins_feats, gaussian_pos, num_iterations=5, num_init_clusters=128, verbose=False):
    similarity_threshold = 0.25
    resolution = 32
    normalized_ins_feats = normalize_vector(ins_feats)
    if verbose:
        logging.info(
            f"Starting over segmentation with {num_init_clusters} initial clusters.")
    over_segmentated_labels = over_segmentation(
        normalize_vector(gaussian_pos),
        normalized_ins_feats,
        num_init_clusters
    )
    new_labels, merged_clusters = create_clusters(
        over_segmentated_labels, normalized_ins_feats, gaussian_pos, similarity_threshold, resolution, verbose=verbose)
    if verbose:
        logging.info(
            f"Initial clustering done with {int(new_labels.max().item() + 1)} clusters.")
    for iteration in range(num_iterations):
        similarity_threshold += 0.25
        if iteration % 2 == 0:
            resolution *= 2
        new_labels, merged_clusters = create_clusters(
            new_labels, normalized_ins_feats, gaussian_pos, similarity_threshold, resolution, verbose=verbose)
        if verbose:
            logging.info(
                f"Iteration {iteration + 1}: Clustering done with resolution {resolution} and similarity threshold {similarity_threshold}.")
            logging.info(
                f"Number of clusters: {int(new_labels.max().item() + 1)}")
    return new_labels, merged_clusters
