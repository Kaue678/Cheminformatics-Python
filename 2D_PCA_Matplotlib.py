from rdkit import Chem
from rdkit.Chem import Descriptors
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def calculate_properties(output_csv, output_sdf):
    """
    Calculates various molecular descriptors for molecules in the 'repellent_library.sdf' file and saves the results in CSV and SDF files.

    Args:
        output_csv: Path to the output CSV file.
        output_sdf: Path to the output SDF file.
    """
    sdf_file = "repellent_library.sdf"
    suppl = Chem.SDMolSupplier(sdf_file)
    writer_sdf = Chem.SDWriter(output_sdf)

    # Create a list to store the data for CSV output
    data = []

    descriptors = []  # List to store molecular descriptors for PCA

    for mol in suppl:
        if mol is None:
            continue

        # Calculate properties
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hba = Descriptors.NumHAcceptors(mol)
        hbd = Descriptors.NumHDonors(mol)
        rot_bonds = Descriptors.NumRotatableBonds(mol)
        num_aromatic_rings = Descriptors.NumAromaticRings(mol)
        num_oxygen_atoms = sum([1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 8])
        num_nitrogen_atoms = sum([1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 7])
        topological_surface_area = Descriptors.TPSA(mol)
        fraction_sp3_carbons = Descriptors.FractionCSP3(mol)

        # Add properties as data fields to the molecule
        mol.SetProp("_MolWt", str(mw))
        mol.SetProp("_logP", str(logp))
        mol.SetProp("_HBA", str(hba))
        mol.SetProp("_HBD", str(hbd))
        mol.SetProp("_RTB", str(rot_bonds))
        mol.SetProp("_AR", str(num_aromatic_rings))
        mol.SetProp("_O", str(num_oxygen_atoms))
        mol.SetProp("_N", str(num_nitrogen_atoms))
        mol.SetProp("_tPSA", str(topological_surface_area))
        mol.SetProp("_FSp3", str(fraction_sp3_carbons))

        # Write molecule to SDF file
        writer_sdf.write(mol)

        # Append data to the list for CSV output
        data.append([mol.GetProp("_Name"), mw,logp, hba, hbd, rot_bonds, num_aromatic_rings, num_oxygen_atoms, num_nitrogen_atoms, topological_surface_area, fraction_sp3_carbons])
        
        # Append descriptors to the list for PCA
        descriptors.append([mw,logp, hba, hbd, rot_bonds, num_aromatic_rings, num_oxygen_atoms, num_nitrogen_atoms, topological_surface_area, fraction_sp3_carbons])

    writer_sdf.close()

    # Write data to CSV file
    with open(output_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Name', 'MolWt','logP', 'HBA', 'HBD', 'RTB', 'AR', 'O', 'N', 'tPSA', 'FSp3'])
        csv_writer.writerows(data)

    # Convert descriptors to numpy array for PCA
    descriptors = np.array(descriptors)

    # Perform PCA
    pca = PCA(n_components=2)
    descriptors_pca = pca.fit_transform(descriptors)

    # Plot PCA
    plt.figure(figsize=(8, 6))
    plt.scatter(descriptors_pca[:, 0], descriptors_pca[:, 1])
    plt.title('PCA of Molecular Descriptors')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.savefig("pca_plot.png", format='png')
    plt.show()

# Example usage
output_csv = "output.csv"
output_sdf = "output.sdf"
calculate_properties(output_csv, output_sdf)

