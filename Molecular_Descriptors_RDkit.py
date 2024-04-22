from rdkit import Chem
from rdkit.Chem import Descriptors
import csv

def calculate_properties(molecular_descriptors_csv, molecular_descriptors_sdf):
    """
    Calculates various molecular descriptors for molecules in the 'repellent_library.sdf' file and saves the results in CSV and SDF files.

    Args:
        output_csv: Path to the output CSV file.
        output_sdf: Path to the output SDF file.
    """
    sdf_file = "repellent_library.sdf"
    suppl = Chem.SDMolSupplier(sdf_file)
    writer_sdf = Chem.SDWriter(molecular_descriptors_sdf)

    # Create a list to store the data for CSV output
    data = []

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

        # Calculate LogD at pH 7.4
        logd_74 = logp - 0.4 * mol.GetNumAtoms()

        # Add properties as data fields to the molecule
        mol.SetProp("_MolWt", str(mw))
        mol.SetProp("_logP", str(logp))
        mol.SetProp("_logD_7.4", str(logd_74))
        mol.SetProp("_HBA", str(hba))
        mol.SetProp("_HBD", str(hbd))
        mol.SetProp("_NumRotBonds", str(rot_bonds))
        mol.SetProp("_NumAromaticRings", str(num_aromatic_rings))
        mol.SetProp("_NumOxygenAtoms", str(num_oxygen_atoms))
        mol.SetProp("_NumNitrogenAtoms", str(num_nitrogen_atoms))
        mol.SetProp("_TopologicalSurfaceArea", str(topological_surface_area))
        mol.SetProp("_FractionSP3Carbons", str(fraction_sp3_carbons))

        # Write molecule to SDF file
        writer_sdf.write(mol)

        # Append data to the list for CSV output
        data.append([mol.GetProp("_Name"), mw, logp, logd_74, hba, hbd, rot_bonds, num_aromatic_rings, num_oxygen_atoms, num_nitrogen_atoms, topological_surface_area, fraction_sp3_carbons])

    writer_sdf.close()

    # Write data to CSV file
    with open(molecular_descriptors_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Name', 'MolWt', 'logP', 'logD_7.4', 'HBA', 'HBD', 'NumRotBonds', 'NumAromaticRings', 'NumOxygenAtoms', 'NumNitrogenAtoms', 'TopologicalSurfaceArea', 'FractionSP3Carbons'])
        csv_writer.writerows(data)

# Example usage
molecular_descriptors_csv = "molecular_descriptors.csv"
molecular_descriptors_sdf = "molecular_descriptors.sdf"
calculate_properties(molecular_descriptors_csv, molecular_descriptors_sdf)
