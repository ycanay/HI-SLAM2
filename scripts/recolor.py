import math
import torch
from tqdm import tqdm
from matplotlib import pyplot as plt
import cv2
import numpy as np
from pathlib import Path
import json
import sys
sys.path.insert(0, str(Path.cwd().parent))


def distinct_colors(K: int) -> list[torch.Tensor]:
    color_div = K ** (1/3)
    steps = 256 / color_div
    colors = []
    last_color = torch.tensor([-steps, 0., 0.])
    for _ in range(K):
        r = last_color[0] + steps
        g = last_color[1]
        b = last_color[2]
        if r >= 256:
            r = r % 256
            g = last_color[1] + steps
            if g >= 256:
                g = g % 256
                b = last_color[2] + steps
        colors.append(torch.tensor(
            [int(math.floor(r)), int(math.floor(g)), int(math.floor(b))]))
        last_color = torch.tensor([r, g, b])
    return colors


def visualize_instance_colors(image, num_clusters, instance_colors):
    unique_ids = np.array(list(range(num_clusters)))
    color_map = {id.item(): instance_colors[i]
                 for i, id in enumerate(unique_ids)}
    instance_ids_np = image.astype(np.uint8)
    instance_ids_img = np.zeros((*instance_ids_np.shape, 3), dtype=np.uint8)
    for instance_id, color in color_map.items():
        mask = instance_ids_np == instance_id
        instance_ids_img[mask] = color
    plt.imshow(instance_ids_img)


def visualize_semantic_colors(image, cluster_data, labels, label_colors):
    label_to_color_map = dict(zip(labels, label_colors))
    instance_to_label_color_map = {id: label_to_color_map[data['label']]
                                   for id, data in cluster_data.items()}
    instance_ids_np = image.astype(np.uint8)
    semantic_img = np.zeros((*instance_ids_np.shape, 3), dtype=np.uint8)
    for instance_id, color in instance_to_label_color_map.items():
        mask = instance_ids_np == int(instance_id)
        semantic_img[mask] = color
    plt.imshow(semantic_img)


run_path = Path("/storage/user/ayu/repos/HI-SLAM2/outputs/semantic/room0_118")
cluster_features_path = run_path / "cluster_features.json"
renders_dir = run_path / "renders"
cluster_renders = renders_dir / "cluster_after_opt"

recolored_instance_save_dir = renders_dir / "recolored_after_opt"
recolored_instance_save_dir.mkdir(parents=True, exist_ok=True)

semantic_save_dir = renders_dir / "semantic_after_opt"
semantic_save_dir.mkdir(parents=True, exist_ok=True)

renders = sorted(list(cluster_renders.glob("*.png")))
image = cv2.cvtColor(cv2.imread(str(renders[0])), cv2.COLOR_RGB2GRAY)

cluster_data = json.load(open(cluster_features_path, 'r'))
num_clusters = len(cluster_data)
labels = set()
for cluster_id, cluster_info in cluster_data.items():
    labels.add(cluster_info['label'])
label_colors = distinct_colors(len(labels))

instance_colors = distinct_colors(num_clusters)


for render_path in tqdm(renders, desc="Visualizing colors", unit="image", total=len(renders)):
    image = cv2.cvtColor(cv2.imread(str(render_path)), cv2.COLOR_RGB2GRAY)

    plt.figure(figsize=(10, 10))
    visualize_instance_colors(image, num_clusters, instance_colors)
    plt.axis('off')
    save_path = recolored_instance_save_dir / render_path.name
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    plt.figure(figsize=(10, 10))
    visualize_semantic_colors(image, cluster_data, labels, label_colors)
    plt.axis('off')
    save_path = semantic_save_dir / render_path.name
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()
