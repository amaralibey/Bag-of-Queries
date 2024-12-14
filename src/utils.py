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
from rich.tree import Tree
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


def display_datasets_stats(datamodule):
    console = Console()
    console.print("\n")

    # Training dataset stats
    train_dataset = datamodule.train_dataset
    train_table = Table(box=None, show_header=False)
    train_table.add_column("Setting", justify="left", no_wrap=True)
    train_table.add_column("Value", style="green")

    train_table.add_row("Number of cities", str(len(train_dataset.cities)))
    train_table.add_row("Number of places", str(len(train_dataset)))
    train_table.add_row("Number of images", str(train_dataset.total_nb_images))

    train_panel = Panel(train_table, title=f"[bold]Training Dataset Stats[/bold]", padding=(1, 2), expand=False)
    console.print(train_panel)
    
    # Training configuration
    config_table = Table(title=None, title_justify="center", box=None, show_header=False)
    config_table.add_column("Setting", justify="left", no_wrap=True)
    config_table.add_column("Value", style="green")

    config_table.add_row("Iterations per epoch", str(len(datamodule.train_dataset) // datamodule.batch_size))
    config_table.add_row("Train batch size (PxK)", f"{datamodule.batch_size}x{datamodule.img_per_place}")
    config_table.add_row("Training image size", f"{datamodule.train_img_size[0]}x{datamodule.train_img_size[1]}")
    config_table.add_row("Validation image size", f"{datamodule.val_img_size[0]}x{datamodule.val_img_size[1]}")
    config_panel = Panel(config_table, title=f"[bold]Training Configuration[/bold]", padding=(1, 2), expand=False)
    console.print(config_panel)
    
    # Validation datasets stats
    val_tree = Tree("Validation Datasets", hide_root=True)
    for i, val_set in enumerate(datamodule.val_datasets):
        val_branch = val_tree.add(f"{val_set.dataset_name}")
        val_branch.add(f"Queries    [green]{val_set.num_queries}[/green]")
        val_branch.add(f"References [green]{val_set.num_references}[/green]")
        
    tree_panel = Panel(val_tree, title=f"[bold]Validation Datasets[/bold]", padding=(1, 2), expand=False)
    
    console.print(tree_panel)
