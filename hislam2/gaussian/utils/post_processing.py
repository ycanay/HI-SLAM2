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

    voxel_clusters = [[] for _ in range(len(unique_voxels))]
    for i, lbl in zip(inverse_indices.tolist(), labels.tolist()):
        voxel_clusters[i].append(lbl)
    voxel_clusters = [torch.tensor(v, dtype=torch.long, device=device).unique()
                      for v in voxel_clusters]

    K = int(labels.max().item()) + 1
    avg_features = [clusters[i]['mean'].to(device) if isinstance(clusters[i]['mean'], torch.Tensor)
                    else torch.tensor(clusters[i]['mean'], dtype=torch.float32, device=device)
                    for i in range(K)]

    adj_matrix = torch.zeros((K, K), dtype=torch.uint8, device=device)
    neighbor_offsets = torch.tensor(
        [o for o in product([-1, 0, 1], repeat=3) if o != (0, 0, 0)],
        dtype=torch.int64,
        device=device
    )

    def is_similar(c1, c2):
        return torch.norm(avg_features[c1] - avg_features[c2], p=2) < similarity_threshold
    for idx, cluster_set in enumerate(voxel_clusters):
        # Same voxel connections
        for i in range(len(cluster_set)):
            for j in range(i + 1, len(cluster_set)):
                c1, c2 = cluster_set[i].item(), cluster_set[j].item()
                if is_similar(c1, c2):
                    adj_matrix[c1, c2] = 1
                    adj_matrix[c2, c1] = 1

        voxel_pos = voxel_coords[idx]
        neighbors = voxel_pos + neighbor_offsets
        for npos in neighbors:
            mask = (voxel_coords == npos).all(dim=1)
            if mask.any():
                neighbor_idx = mask.nonzero(as_tuple=True)[0].item()
                for c1 in cluster_set:
                    for c2 in voxel_clusters[neighbor_idx]:
                        if c1 != c2 and is_similar(c1.item(), c2.item()):
                            adj_matrix[c1, c2] = 1
                            adj_matrix[c2, c1] = 1

    return adj_matrix


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
        merged_features = torch.cat(
            [clusters[i]['mean'] for i in comp], dim=0)
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
    new_labels, _ = create_clusters(
        over_segmentated_labels, normalized_ins_feats, gaussian_pos, similarity_threshold, resolution)
    if verbose:
        logging.info(
            f"Initial clustering done with {new_labels.max().item() + 1} clusters.")
    for iteration in range(num_iterations):
        similarity_threshold += 0.25
        if iteration % 2 == 0:
            resolution *= 2
        new_labels, _ = create_clusters(
            new_labels, normalized_ins_feats, gaussian_pos, similarity_threshold, resolution)
        if verbose:
            logging.info(
                f"Iteration {iteration + 1}: Clustering done with resolution {resolution} and similarity threshold {similarity_threshold}.")
            logging.info(f"Number of clusters: {new_labels.max().item() + 1}")
    return new_labels
