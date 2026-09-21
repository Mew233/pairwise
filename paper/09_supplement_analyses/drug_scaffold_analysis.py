"""
Drug Scaffold Analysis for Rebuttal
Extract SMILES, compute Bemis-Murcko scaffolds, and calculate Tanimoto similarities.
"""
import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict
import pickle

# RDKit imports
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import DataStructs
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

# Setup paths
HERE = os.path.dirname(os.path.abspath(__file__))

# Output directories
DATA_DIR = os.path.join(HERE, 'data')
FIG_DIR = os.path.join(HERE, 'figures')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


def extract_drug_smiles_from_p13():
    """
    Extract unique drugs and their SMILES from p13 dataset.
    Drug IDs are in compound0_x and compound0_y columns (DB format).
    """
    print("Loading p13 trueset data...")
    p13_path = os.path.join(ROOT_DIR, 'data', 'synergy_data', 'p13', 'p13_trueset.csv')
    df = pd.read_csv(p13_path)

    # Extract drug IDs
    drugs_x = df[['compound0_x', 'pubchemID_x']].rename(
        columns={'compound0_x': 'drug_db_id', 'pubchemID_x': 'pubchem_id'}
    ).drop_duplicates()

    drugs_y = df[['compound0_y', 'pubchemID_y']].rename(
        columns={'compound0_y': 'drug_db_id', 'pubchemID_y': 'pubchem_id'}
    ).drop_duplicates()

    all_drugs = pd.concat([drugs_x, drugs_y], ignore_index=True).drop_duplicates()

    print(f"Found {len(all_drugs)} unique drugs")

    # Now we need to get SMILES - check if there's a structures file
    # Try to find SMILES in existing drug data
    drug_data_dir = os.path.join(ROOT_DIR, 'data', 'drug_data')

    # Check for structures.sdf mentioned in README
    sdf_path = os.path.join(ROOT_DIR, 'data', 'structures.sdf')
    if os.path.exists(sdf_path):
        print(f"Found structures.sdf at {sdf_path}")
        smiles_dict = parse_sdf_to_smiles(sdf_path)
    else:
        print("Warning: structures.sdf not found. Will need to fetch from PubChem.")
        smiles_dict = {}

    # Map drugs to SMILES (with progress bar)
    print(f"Fetching SMILES for {len(all_drugs)} drugs from PubChem...")
    from tqdm import tqdm
    tqdm.pandas()
    all_drugs['smiles'] = all_drugs.progress_apply(
        lambda row: get_smiles_for_drug(row, smiles_dict), axis=1
    )

    # Filter out drugs without SMILES
    all_drugs_with_smiles = all_drugs.dropna(subset=['smiles'])
    print(f"Successfully obtained SMILES for {len(all_drugs_with_smiles)}/{len(all_drugs)} drugs")

    return all_drugs_with_smiles


def parse_sdf_to_smiles(sdf_path):
    """Parse SDF file and extract SMILES for each compound."""
    print(f"Parsing {sdf_path}...")
    smiles_dict = {}
    suppl = Chem.SDMolSupplier(sdf_path)

    for mol in suppl:
        if mol is not None:
            # Try to get PubChem ID from properties
            if mol.HasProp('PUBCHEM_COMPOUND_CID'):
                pubchem_id = mol.GetProp('PUBCHEM_COMPOUND_CID')
                smiles = Chem.MolToSmiles(mol)
                smiles_dict[pubchem_id] = smiles

    print(f"Extracted SMILES for {len(smiles_dict)} compounds from SDF")
    return smiles_dict


def get_smiles_for_drug(row, smiles_dict):
    """Get SMILES for a drug from various sources."""
    import re

    # Try PubChem ID in dict
    pubchem_str = str(row['pubchem_id'])
    if pubchem_str in smiles_dict:
        return smiles_dict[pubchem_str]

    # Try fetching from PubChem
    pubchem_id = row['pubchem_id']
    if pd.notna(pubchem_id):
        # Extract numeric ID from "Compound(12345)" format
        if isinstance(pubchem_id, str):
            match = re.search(r'\d+', pubchem_id)
            if match:
                pubchem_id = int(match.group())
            else:
                return None
        return fetch_smiles_from_pubchem(int(pubchem_id))

    return None


def fetch_smiles_from_pubchem(pubchem_id, max_retries=3):
    """Fetch SMILES from PubChem API (use sparingly to avoid rate limits)."""
    import time
    import requests

    for attempt in range(max_retries):
        try:
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{pubchem_id}/property/CanonicalSMILES/JSON"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return data['PropertyTable']['Properties'][0]['CanonicalSMILES']
            time.sleep(0.5)  # Rate limiting
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                print(f"Failed to fetch SMILES for PubChem ID {pubchem_id}: {e}")
    return None


def compute_scaffolds(drugs_df):
    """Compute Bemis-Murcko scaffolds for all drugs."""
    print("\nComputing Bemis-Murcko scaffolds...")

    scaffolds = []
    scaffold_mols = []

    for idx, row in drugs_df.iterrows():
        mol = Chem.MolFromSmiles(row['smiles'])
        if mol is not None:
            try:
                scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                scaffold_smiles = Chem.MolToSmiles(scaffold)
                scaffolds.append(scaffold_smiles)
                scaffold_mols.append(scaffold)
            except:
                scaffolds.append(None)
                scaffold_mols.append(None)
        else:
            scaffolds.append(None)
            scaffold_mols.append(None)

    drugs_df['scaffold_smiles'] = scaffolds

    # Assign scaffold clusters
    scaffold_to_id = {}
    scaffold_ids = []
    next_id = 0

    for scaffold in scaffolds:
        if scaffold is None:
            scaffold_ids.append(-1)
        elif scaffold not in scaffold_to_id:
            scaffold_to_id[scaffold] = next_id
            scaffold_ids.append(next_id)
            next_id += 1
        else:
            scaffold_ids.append(scaffold_to_id[scaffold])

    drugs_df['scaffold_cluster_id'] = scaffold_ids

    print(f"Found {len(scaffold_to_id)} unique scaffolds")
    print(f"Scaffold cluster sizes:")
    cluster_sizes = drugs_df['scaffold_cluster_id'].value_counts().sort_values(ascending=False)
    print(cluster_sizes.head(10))

    return drugs_df


def compute_tanimoto_matrix(drugs_df):
    """Compute pairwise Tanimoto similarity matrix using Morgan fingerprints."""
    print("\nComputing pairwise Tanimoto similarities...")

    # Generate Morgan fingerprints
    fps = []
    valid_indices = []

    for idx, row in drugs_df.iterrows():
        mol = Chem.MolFromSmiles(row['smiles'])
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            fps.append(fp)
            valid_indices.append(idx)

    print(f"Generated fingerprints for {len(fps)} drugs")

    # Compute pairwise Tanimoto similarities
    n = len(fps)
    tanimoto_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            similarity = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            tanimoto_matrix[i, j] = similarity
            tanimoto_matrix[j, i] = similarity

    # Create mapping from original index to matrix index
    idx_mapping = {orig_idx: mat_idx for mat_idx, orig_idx in enumerate(valid_indices)}

    return tanimoto_matrix, idx_mapping, valid_indices


def plot_similarity_histogram(tanimoto_matrix):
    """Plot histogram of pairwise Tanimoto similarities."""
    print("\nGenerating similarity histogram...")

    # Extract upper triangle (excluding diagonal)
    n = tanimoto_matrix.shape[0]
    similarities = []
    for i in range(n):
        for j in range(i+1, n):
            similarities.append(tanimoto_matrix[i, j])

    similarities = np.array(similarities)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.hist(similarities, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Tanimoto Similarity', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Pairwise Drug Tanimoto Similarity Distribution', fontsize=14)
    plt.axvline(x=0.85, color='red', linestyle='--', linewidth=2, label='Threshold = 0.85')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Add statistics
    mean_sim = np.mean(similarities)
    median_sim = np.median(similarities)
    high_sim = np.sum(similarities > 0.85)
    total_pairs = len(similarities)

    stats_text = f'Mean: {mean_sim:.3f}\nMedian: {median_sim:.3f}\n>0.85: {high_sim}/{total_pairs} ({100*high_sim/total_pairs:.2f}%)'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    output_path = os.path.join(FIG_DIR, 'drug_similarity_histogram.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"Saved histogram to {output_path}")

    # Print near-duplicate statistics
    print(f"\nSimilarity statistics:")
    print(f"  Mean similarity: {mean_sim:.3f}")
    print(f"  Median similarity: {median_sim:.3f}")
    print(f"  Pairs with Tanimoto > 0.85: {high_sim}/{total_pairs} ({100*high_sim/total_pairs:.2f}%)")
    print(f"  Pairs with Tanimoto > 0.90: {np.sum(similarities > 0.90)}")
    print(f"  Pairs with Tanimoto > 0.95: {np.sum(similarities > 0.95)}")


def plot_scaffold_dendrogram(tanimoto_matrix, drugs_df, valid_indices):
    """Plot hierarchical clustering dendrogram based on scaffolds."""
    print("\nGenerating scaffold dendrogram...")

    # Convert similarity to distance
    distance_matrix = 1 - tanimoto_matrix

    # Convert to condensed distance matrix for linkage
    condensed_dist = squareform(distance_matrix, checks=False)

    # Perform hierarchical clustering
    linkage_matrix = linkage(condensed_dist, method='average')

    # Plot dendrogram
    plt.figure(figsize=(15, 8))
    dendrogram(linkage_matrix, no_labels=True)
    plt.xlabel('Drug Index', fontsize=12)
    plt.ylabel('Distance (1 - Tanimoto)', fontsize=12)
    plt.title('Hierarchical Clustering of Drugs by Structural Similarity', fontsize=14)
    plt.axhline(y=0.15, color='red', linestyle='--', linewidth=2, label='Cutoff = 0.15 (Tanimoto = 0.85)')
    plt.legend()
    plt.tight_layout()

    output_path = os.path.join(FIG_DIR, 'drug_scaffold_dendrogram.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"Saved dendrogram to {output_path}")


def identify_near_duplicates(tanimoto_matrix, drugs_df, valid_indices, threshold=0.85):
    """Identify and report near-duplicate drug pairs."""
    print(f"\nIdentifying near-duplicate pairs (Tanimoto > {threshold})...")

    near_duplicates = []
    n = tanimoto_matrix.shape[0]

    for i in range(n):
        for j in range(i+1, n):
            if tanimoto_matrix[i, j] > threshold:
                idx_i = valid_indices[i]
                idx_j = valid_indices[j]
                near_duplicates.append({
                    'drug1_db_id': drugs_df.loc[idx_i, 'drug_db_id'],
                    'drug2_db_id': drugs_df.loc[idx_j, 'drug_db_id'],
                    'drug1_pubchem': drugs_df.loc[idx_i, 'pubchem_id'],
                    'drug2_pubchem': drugs_df.loc[idx_j, 'pubchem_id'],
                    'tanimoto_similarity': tanimoto_matrix[i, j]
                })

    near_dup_df = pd.DataFrame(near_duplicates)
    near_dup_df = near_dup_df.sort_values('tanimoto_similarity', ascending=False)

    print(f"Found {len(near_dup_df)} near-duplicate pairs")
    if len(near_dup_df) > 0:
        print("\nTop 10 most similar pairs:")
        print(near_dup_df.head(10).to_string(index=False))

    # Save to file
    output_path = os.path.join(DATA_DIR, 'near_duplicate_drugs.csv')
    near_dup_df.to_csv(output_path, index=False)
    print(f"\nSaved near-duplicate pairs to {output_path}")

    return near_dup_df


def main():
    print("="*80)
    print("DRUG SCAFFOLD ANALYSIS FOR REBUTTAL")
    print("="*80)

    # Step 1: Extract drugs and SMILES
    drugs_df = extract_drug_smiles_from_p13()

    if len(drugs_df) == 0:
        print("ERROR: No drugs with SMILES found. Cannot proceed.")
        print("Please ensure structures.sdf is available or enable PubChem fetching.")
        return

    # Save drugs with SMILES
    output_path = os.path.join(DATA_DIR, 'drug_smiles.csv')
    drugs_df.to_csv(output_path, index=False)
    print(f"\nSaved drug SMILES to {output_path}")

    # Step 2: Compute scaffolds
    drugs_df = compute_scaffolds(drugs_df)

    # Save scaffolds
    scaffold_path = os.path.join(DATA_DIR, 'drug_scaffolds.csv')
    drugs_df[['drug_db_id', 'pubchem_id', 'scaffold_smiles', 'scaffold_cluster_id']].to_csv(
        scaffold_path, index=False
    )
    print(f"Saved scaffold data to {scaffold_path}")

    # Step 3: Compute Tanimoto similarities
    tanimoto_matrix, idx_mapping, valid_indices = compute_tanimoto_matrix(drugs_df)

    # Save Tanimoto matrix
    matrix_path = os.path.join(DATA_DIR, 'drug_tanimoto_matrix.npy')
    np.save(matrix_path, tanimoto_matrix)
    print(f"Saved Tanimoto matrix to {matrix_path}")

    # Save index mapping
    mapping_path = os.path.join(DATA_DIR, 'drug_tanimoto_index_mapping.pkl')
    with open(mapping_path, 'wb') as f:
        pickle.dump({'idx_mapping': idx_mapping, 'valid_indices': valid_indices}, f)
    print(f"Saved index mapping to {mapping_path}")

    # Step 4: Generate visualizations
    plot_similarity_histogram(tanimoto_matrix)
    plot_scaffold_dendrogram(tanimoto_matrix, drugs_df, valid_indices)

    # Step 5: Identify near-duplicates
    near_dup_df = identify_near_duplicates(tanimoto_matrix, drugs_df, valid_indices, threshold=0.85)

    print("\n" + "="*80)
    print("DRUG SCAFFOLD ANALYSIS COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
