# ----------------------------------------------------------------------------
# Copyright (c) 2024 Amar Ali-bey
#
# https://github.com/amaralibey/Bag-of-Queries
#
# See LICENSE file in the project root.
# ----------------------------------------------------------------------------

from typing import List, Dict, Tuple

import numpy as np
import faiss

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

def compute_recall_performance(
                descriptors,
                num_references,
                num_queries,
                ground_truth,
                k_values=[1, 5, 10],
    ):
    """
    Compute recall@K scores for a given dataset and descriptors using FAISS.

    Parameters
    ----------
    descriptors : numpy.ndarray or torch.Tensor
        descriptors of both reference and query images. Shape is (num_images, embedding_dim).
        Note that the first `num_references` descriptors are reference images (num_images = num_references + num_queries).
    num_references : int
        Number of reference images.
    num_queries : int
        Number of query images.
    ground_truth : list of lists
        Ground truth labels for each query image. Each list contains the indices of relevant reference images.
    k_values : list, optional
        List of 'K' values for which recall@K scores will be computed, by default [1, 5, 10].

    Returns
    -------
    dict
        A dictionary mapping each 'K' value to its corresponding recall@K score.
    """

    assert num_references + num_queries == len(
        descriptors
    ), "Number of references and queries do not match the number of descriptors. THERE IS A BUG!"

    embed_size = descriptors.shape[1]
    faiss_index = faiss.IndexFlatL2(embed_size)
    
    # add references
    faiss_index.add(descriptors[:num_references])

    # search for queries in the index
    _, predictions = faiss_index.search(descriptors[num_references:], max(k_values))

    # start calculating recall_at_k
    correct_at_k = np.zeros(len(k_values))
    for q_idx, pred in enumerate(predictions):
        for i, n in enumerate(k_values):
            # if in top N then also in top NN, where NN > N
            if np.any(np.in1d(pred[:n], ground_truth[q_idx])):
                correct_at_k[i:] += 1
                break

    correct_at_k = correct_at_k / len(predictions)
    d = {k: v for (k, v) in zip(k_values, correct_at_k)}
    return d


def display_recall_performance(recalls_list: List[Dict[int, float]], 
                                val_set_names: List[str], 
                                title: str = "Recall@k Performance") -> None:
    if not recalls_list:
        return
    console = Console()
    console.print("\n")
    table = Table(title=None, box=box.SIMPLE, header_style="bold")
    k_values = list(recalls_list[0].keys())

    table.add_column("Dataset", justify="left")
    for k in k_values:
        table.add_column(f"R@{k}", justify="center")

    for i, recalls in enumerate(recalls_list):
        table.add_row(val_set_names[i], *[f"{100 * v:.2f}" for v in recalls.values()])

    console.print(Panel(table, expand=False, title=title))
    console.print("\n")