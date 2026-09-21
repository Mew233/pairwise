"""
Cell Lineage Analysis for Rebuttal
Compute cell line expression distances and identify similar cell lines.
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt
import seaborn as sns

# Setup paths
from pairwise.paths import RESULTS_DIR as PW_RESULTS

HERE = os.path.dirname(os.path.abspath(__file__))

# Output directories
DATA_DIR = os.path.join(HERE, 'data')
FIG_DIR = os.path.join(HERE, 'figures')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


def load_cell_expression_data():
    """Load CCLE expression data for cells in the processed synergy dataset."""
    print("Loading processed synergy data...")
    synergy_path = os.path.join(PW_RESULTS, 'processed_synergydf_p13.csv')
    synergy_df = pd.read_csv(synergy_path, index_col=0)

    # Get unique cells
    unique_cells = synergy_df['cell'].unique()
    print(f"Found {len(unique_cells)} unique cell lines in synergy data")

    # Load CCLE expression data
    print("\nLoading CCLE expression data...")
    ccle_exp_path = os.path.join(ROOT_DIR, 'data', 'cell_line_data', 'CCLE', 'CCLE_exp.csv')

    if not os.path.exists(ccle_exp_path):
        print(f"ERROR: CCLE expression data not found at {ccle_exp_path}")
        return None, None

    ccle_exp = pd.read_csv(ccle_exp_path, index_col=0)
    print(f"CCLE expression data shape: {ccle_exp.shape}")
    print(f"Genes: {ccle_exp.shape[0]}, Cell lines: {ccle_exp.shape[1]}")

    # Filter to cells in synergy dataset
    available_cells = [c for c in unique_cells if c in ccle_exp.columns]
    print(f"Found expression data for {len(available_cells)}/{len(unique_cells)} cells")

    # Extract expression matrix (genes x cells) -> transpose to (cells x genes)
    cell_exp_matrix = ccle_exp[available_cells].T

    # Get tissue information
    cell_tissue_map = synergy_df[['cell', 'tissue_name']].drop_duplicates().set_index('cell')['tissue_name'].to_dict()
    tissues = [cell_tissue_map.get(c, 'unknown') for c in available_cells]

    # Create cell info dataframe
    cell_info = pd.DataFrame({
        'cell_id': available_cells,
        'tissue': tissues
    })

    return cell_exp_matrix, cell_info


def compute_expression_distance_matrix(cell_exp_matrix, metric='correlation'):
    """
    Compute pairwise distance matrix between cell lines.

    Args:
        cell_exp_matrix: DataFrame with cells as rows, genes as columns
        metric: 'correlation' or 'euclidean'
    """
    print(f"\nComputing pairwise cell distances (metric: {metric})...")

    if metric == 'correlation':
        # Compute correlation matrix, then convert to distance
        corr_matrix = np.corrcoef(cell_exp_matrix.values)
        distance_matrix = 1 - corr_matrix  # Convert correlation to distance
        distance_matrix = np.clip(distance_matrix, 0, 2)  # Clip to [0, 2]
    elif metric == 'euclidean':
        # Standardize features first
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        cell_exp_scaled = scaler.fit_transform(cell_exp_matrix.values)

        # Compute pairwise Euclidean distances
        distances = pdist(cell_exp_scaled, metric='euclidean')
        distance_matrix = squareform(distances)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    print(f"Distance matrix shape: {distance_matrix.shape}")
    print(f"Distance range: [{distance_matrix.min():.3f}, {distance_matrix.max():.3f}]")
    print(f"Mean distance: {distance_matrix.mean():.3f}")

    return distance_matrix


def cluster_cells_hierarchical(distance_matrix, cell_info, n_clusters=None, distance_threshold=None):
    """
    Perform hierarchical clustering on cell lines.

    Args:
        distance_matrix: Pairwise distance matrix
        cell_info: DataFrame with cell information
        n_clusters: Number of clusters (if None, use distance_threshold)
        distance_threshold: Distance threshold for clustering
    """
    print("\nPerforming hierarchical clustering...")

    # Convert to condensed distance matrix
    condensed_dist = squareform(distance_matrix, checks=False)

    # Perform hierarchical clustering
    linkage_matrix = linkage(condensed_dist, method='average')

    # Cut tree to form clusters
    if n_clusters is not None:
        cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        print(f"Created {n_clusters} clusters")
    elif distance_threshold is not None:
        cluster_labels = fcluster(linkage_matrix, distance_threshold, criterion='distance')
        n_clusters = len(np.unique(cluster_labels))
        print(f"Created {n_clusters} clusters with distance threshold {distance_threshold}")
    else:
        # Default: use elbow at ~20 clusters
        cluster_labels = fcluster(linkage_matrix, 20, criterion='maxclust')
        n_clusters = 20
        print(f"Created {n_clusters} clusters (default)")

    # Add cluster labels to cell_info
    cell_info['cluster_id'] = cluster_labels

    # Print cluster statistics
    print("\nCluster size distribution:")
    cluster_sizes = cell_info['cluster_id'].value_counts().sort_index()
    print(cluster_sizes.head(20))

    return linkage_matrix, cluster_labels, cell_info


def plot_cell_dendrogram(linkage_matrix, cell_info):
    """Plot hierarchical clustering dendrogram for cell lines."""
    print("\nGenerating cell line dendrogram...")

    plt.figure(figsize=(20, 8))

    # Create color map for tissues
    unique_tissues = cell_info['tissue'].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_tissues)))
    tissue_color_map = dict(zip(unique_tissues, colors))

    # Create leaf colors based on tissue
    leaf_colors = [tissue_color_map[tissue] for tissue in cell_info['tissue']]

    # Plot dendrogram
    dendrogram(linkage_matrix, no_labels=True, leaf_rotation=90)

    plt.xlabel('Cell Line Index', fontsize=12)
    plt.ylabel('Distance', fontsize=12)
    plt.title('Hierarchical Clustering of Cell Lines by Expression Profile', fontsize=14)
    plt.tight_layout()

    output_path = os.path.join(FIG_DIR, 'cell_dendrogram.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"Saved dendrogram to {output_path}")
    plt.close()


def plot_distance_distribution(distance_matrix):
    """Plot distribution of pairwise cell distances."""
    print("\nGenerating distance distribution plot...")

    # Extract upper triangle (excluding diagonal)
    n = distance_matrix.shape[0]
    distances = []
    for i in range(n):
        for j in range(i+1, n):
            distances.append(distance_matrix[i, j])

    distances = np.array(distances)

    plt.figure(figsize=(10, 6))
    plt.hist(distances, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Expression Distance (1 - Correlation)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Pairwise Cell Line Expression Distance Distribution', fontsize=14)

    # Add statistics
    mean_dist = np.mean(distances)
    median_dist = np.median(distances)

    stats_text = f'Mean: {mean_dist:.3f}\nMedian: {median_dist:.3f}'
    plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = os.path.join(FIG_DIR, 'cell_distance_distribution.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"Saved distance distribution to {output_path}")
    plt.close()


def plot_umap_projection(cell_exp_matrix, cell_info):
    """Generate UMAP projection of cell lines colored by tissue."""
    print("\nGenerating UMAP projection...")

    try:
        import umap
        from sklearn.preprocessing import StandardScaler

        # Standardize features
        scaler = StandardScaler()
        cell_exp_scaled = scaler.fit_transform(cell_exp_matrix.values)

        # Compute UMAP
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='correlation', random_state=42)
        embedding = reducer.fit_transform(cell_exp_scaled)

        # Plot
        plt.figure(figsize=(12, 10))

        # Get unique tissues and create color map
        unique_tissues = sorted(cell_info['tissue'].unique())
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_tissues)))

        for i, tissue in enumerate(unique_tissues):
            mask = cell_info['tissue'] == tissue
            plt.scatter(embedding[mask, 0], embedding[mask, 1],
                       c=[colors[i]], label=tissue, alpha=0.7, s=50)

        plt.xlabel('UMAP 1', fontsize=12)
        plt.ylabel('UMAP 2', fontsize=12)
        plt.title('UMAP Projection of Cell Lines (colored by tissue)', fontsize=14)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.tight_layout()

        output_path = os.path.join(FIG_DIR, 'cell_umap.pdf')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
        print(f"Saved UMAP to {output_path}")
        plt.close()

    except ImportError:
        print("WARNING: umap-learn not installed. Skipping UMAP plot.")
        print("Install with: pip install umap-learn")


def plot_distance_heatmap(distance_matrix, cell_info, max_cells=100):
    """Plot heatmap of cell-cell distances (subsample if too large)."""
    print("\nGenerating distance heatmap...")

    n_cells = distance_matrix.shape[0]

    if n_cells > max_cells:
        print(f"Subsampling {max_cells} cells for visualization (total: {n_cells})")
        # Subsample cells, stratified by tissue
        from sklearn.model_selection import train_test_split

        indices = np.arange(n_cells)
        try:
            _, sample_indices = train_test_split(
                indices, test_size=max_cells, stratify=cell_info['tissue'], random_state=42
            )
        except:
            # If stratification fails, just random sample
            sample_indices = np.random.choice(indices, size=max_cells, replace=False)

        sample_indices = sorted(sample_indices)
        dist_sub = distance_matrix[np.ix_(sample_indices, sample_indices)]
        tissue_labels = cell_info.iloc[sample_indices]['tissue'].values
    else:
        dist_sub = distance_matrix
        tissue_labels = cell_info['tissue'].values
        sample_indices = np.arange(n_cells)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot heatmap
    im = ax.imshow(dist_sub, cmap='viridis', aspect='auto')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Distance (1 - Correlation)', fontsize=12)

    ax.set_xlabel('Cell Line Index', fontsize=12)
    ax.set_ylabel('Cell Line Index', fontsize=12)
    ax.set_title(f'Cell Line Expression Distance Matrix (n={len(sample_indices)})', fontsize=14)

    plt.tight_layout()

    output_path = os.path.join(FIG_DIR, 'cell_distance_heatmap.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"Saved heatmap to {output_path}")
    plt.close()


def identify_similar_cell_pairs(distance_matrix, cell_info, threshold=0.2):
    """Identify pairs of cell lines with high similarity (low distance)."""
    print(f"\nIdentifying similar cell pairs (distance < {threshold})...")

    similar_pairs = []
    n = distance_matrix.shape[0]

    for i in range(n):
        for j in range(i+1, n):
            if distance_matrix[i, j] < threshold:
                similar_pairs.append({
                    'cell1': cell_info.iloc[i]['cell_id'],
                    'cell2': cell_info.iloc[j]['cell_id'],
                    'tissue1': cell_info.iloc[i]['tissue'],
                    'tissue2': cell_info.iloc[j]['tissue'],
                    'distance': distance_matrix[i, j],
                    'same_tissue': cell_info.iloc[i]['tissue'] == cell_info.iloc[j]['tissue']
                })

    similar_df = pd.DataFrame(similar_pairs)
    similar_df = similar_df.sort_values('distance')

    print(f"Found {len(similar_df)} similar cell pairs")

    if len(similar_df) > 0:
        print(f"  Same tissue: {similar_df['same_tissue'].sum()}")
        print(f"  Different tissue: {(~similar_df['same_tissue']).sum()}")
        print("\nTop 10 most similar pairs:")
        print(similar_df.head(10).to_string(index=False))

    # Save to file
    output_path = os.path.join(DATA_DIR, 'similar_cell_pairs.csv')
    similar_df.to_csv(output_path, index=False)
    print(f"\nSaved similar cell pairs to {output_path}")

    return similar_df


def main():
    print("="*80)
    print("CELL LINEAGE ANALYSIS FOR REBUTTAL")
    print("="*80)

    # Step 1: Load cell expression data
    cell_exp_matrix, cell_info = load_cell_expression_data()

    if cell_exp_matrix is None:
        print("ERROR: Could not load cell expression data. Exiting.")
        return

    # Step 2: Compute distance matrix
    distance_matrix = compute_expression_distance_matrix(cell_exp_matrix, metric='correlation')

    # Save distance matrix
    matrix_path = os.path.join(DATA_DIR, 'cell_distance_matrix.npy')
    np.save(matrix_path, distance_matrix)
    print(f"Saved distance matrix to {matrix_path}")

    # Step 3: Cluster cells
    linkage_matrix, cluster_labels, cell_info = cluster_cells_hierarchical(
        distance_matrix, cell_info, n_clusters=20
    )

    # Save cell clusters
    cluster_path = os.path.join(DATA_DIR, 'cell_clusters.csv')
    cell_info.to_csv(cluster_path, index=False)
    print(f"Saved cell cluster assignments to {cluster_path}")

    # Step 4: Generate visualizations
    plot_distance_distribution(distance_matrix)
    plot_cell_dendrogram(linkage_matrix, cell_info)
    plot_distance_heatmap(distance_matrix, cell_info, max_cells=100)
    plot_umap_projection(cell_exp_matrix, cell_info)

    # Step 5: Identify similar cell pairs
    similar_df = identify_similar_cell_pairs(distance_matrix, cell_info, threshold=0.2)

    print("\n" + "="*80)
    print("CELL LINEAGE ANALYSIS COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
