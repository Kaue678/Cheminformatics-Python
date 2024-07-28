from rdkit import Chem  
from rdkit.Chem import Descriptors  
import pandas as pd  
import seaborn as sns  
import matplotlib.pyplot as plt  

# Load molecules from output.sdf file  
suppl = Chem.SDMolSupplier("molecular_descriptors.sdf")  
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

# Create DataFrame  
df = pd.DataFrame(data, columns=['MolWt', 'logP', 'HBA', 'HBD', 'RTB', 'AR', 'O', 'N', 'tPSA', 'FSp3'])  

# Calculate Pearson correlation  
pearson_corr = df.corr(method='pearson')  

# Calculate Kendall correlation  
kendall_corr = df.corr(method='kendall')  

# Output results to CSV files  
pearson_corr.to_csv("pearson_correlation.csv")  
kendall_corr.to_csv("kendall_correlation.csv")  

print("Pearson correlation saved as pearson_correlation.csv")  
print("Kendall correlation saved as kendall_correlation.csv")  

# Create heatmap for Kendall correlation  
plt.figure(figsize=(10, 8))  
sns.heatmap(kendall_corr, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})  
plt.title('Kendall Correlation Heatmap')  
plt.savefig("kendall_correlation_heatmap.png")  # Save the figure as PNG  
plt.close()  # Close the figure  
print("Kendall correlation heatmap saved as kendall_correlation_heatmap.png")

# Create heatmap for Pearson correlation  
plt.figure(figsize=(10, 8))  
sns.heatmap(pearson_corr, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})  
plt.title('Pearson Correlation Heatmap')  
plt.savefig("pearson_correlation_heatmap.png")  # Save the figure as PNG  
plt.close()  # Close the figure  
print("Pearson correlation heatmap saved as pearson_correlation_heatmap.png")
