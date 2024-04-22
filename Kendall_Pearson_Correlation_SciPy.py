from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd
from scipy.stats import pearsonr, kendalltau

# Load molecules from output.sdf file
suppl = Chem.SDMolSupplier("output.sdf")
molecules = [mol for mol in suppl if mol]

# Create a DataFrame to store molecular descriptors
data = []
for mol in molecules:
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
    data.append([mw, logp, hba, hbd, rot_bonds, num_aromatic_rings, num_oxygen_atoms, num_nitrogen_atoms, topological_surface_area, fraction_sp3_carbons])

df = pd.DataFrame(data, columns=['MolWt', 'logP', 'HBA', 'HBD', 'NumRotBonds', 'NumAromaticRings', 'NumOxygenAtoms', 'NumNitrogenAtoms', 'TopologicalSurfaceArea', 'FractionSP3Carbons'])

# Calculate Pearson correlation
pearson_corr = df.corr(method='pearson')

# Calculate Kendall correlation
kendall_corr = df.corr(method='kendall')

# Output results to CSV files
pearson_corr.to_csv("pearson_correlation.csv")
kendall_corr.to_csv("kendall_correlation.csv")

print("Pearson correlation saved as pearson_correlation.csv")
print("Kendall correlation saved as kendall_correlation.csv")

