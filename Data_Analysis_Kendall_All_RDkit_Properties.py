"""  
This script processes two repellent libraries: one containing active compounds (hits, labeled as `1`)   
and the other containing inactive compounds (decoys, labeled as `0`). It calculates all molecular   
descriptors available in RDKit for the molecules in both libraries. Afterward, it computes the Kendall   
correlation values between each molecular descriptor and the activity label (`Label`).   

The script identifies the 25 descriptors with the highest absolute Kendall correlation values and   
visualizes them in a horizontal bar plot. Additionally, it saves the full dataset, all Kendall correlation   
values, and the top 25 correlations to CSV files for further analysis.  
"""  

from rdkit import Chem  
from rdkit.Chem import Descriptors, rdMolDescriptors  
from rdkit.ML.Descriptors import MoleculeDescriptors  # Correct import for MoleculeDescriptors  
import pandas as pd  
import seaborn as sns  
import matplotlib.pyplot as plt  
from scipy.stats import spearmanr, kendalltau  


# Load molecule  
def load_molecules(sdf_file):  
    """Load molecules from an SDF file and extract all molecular descriptors."""  
    suppl = Chem.SDMolSupplier(sdf_file)  
    data, name = [], []  
    descriptor_calculator = MoleculeDescriptors.MolecularDescriptorCalculator(  
        [desc[0] for desc in Chem.Descriptors._descList]  # Get the names of all descriptors  
    )  

    for mol in suppl:  
        if mol is None:  
            continue  
        # Calculate all molecular descriptors  
        properties = descriptor_calculator.CalcDescriptors(mol)  
        data.append(properties)  

        name.append(mol.GetProp("Molecule") if mol.HasProp("Molecule") else "Sem Nome")  

    df = pd.DataFrame(data, columns=[desc[0] for desc in Chem.Descriptors._descList])  

    df.insert(0, 'Molecule', name)  

    return df  

# Calculate Kendall correlation  
def calc_corr(df, top):  
    """Calculate Kendall correlation and return all correlations and the top N correlations."""  
    corr = df.iloc[:, 1:-1].corrwith(df['Label'], method='kendall').sort_values(ascending=False).dropna()  

    # Get the top N descriptors  
    top_corr = corr.abs().nlargest(top)  
    corr_top = corr.loc[top_corr.index]  

    return corr, corr_top  


if __name__ == "__main__":  

    # Load molecules from both SDF files  
    df0 = load_molecules("repellent_library_0.sdf")  
    df1 = load_molecules("repellent_library_1.sdf")  

    # Insert labels  
    df1['Label'] = 1  
    df0['Label'] = 0  

    # Concatenate the dataframes  
    df = pd.concat([df0, df1], axis=0)  
  
    # Remove duplicates  
    df = df.drop_duplicates(subset='Molecule', keep='first')  

    # Remove columns with NaN instances  
    df = df.dropna(axis=1, how='any')  

    # Calculate Kendall Correlation  
    # corr_kd is the correlation for all descriptors  
    # corr_kd_top is the correlation of the selection of descriptors based on top values  
    top = 25  # select the threshold for the best descriptors  
    corr_kd, corr_kd_top = calc_corr(df, top)   

    # Plot Kendall Correlation   
    plt.figure(figsize=(10, 8))  
    corr_kd_top.sort_values().plot(kind='barh', color='skyblue', edgecolor='black')  
    plt.title('Top 25 Kendall Correlations with Repellent Activity', fontsize=16)  
    plt.xlabel('Kendall Correlation', fontsize=14)  
    plt.ylabel('Descriptors', fontsize=14)  
    plt.tight_layout()  
    plt.savefig('kendall_correlation_plot.png')  # Save the plot as an image  
    plt.show()  

    # Save data  
    df.to_csv('repellent_descriptors.csv', sep=',', index=False)  
    corr_kd.to_csv('kendall_correlation.csv', sep=',', header=False)  
    corr_kd_top.to_csv('top_25_kendall_correlation.csv', sep=',', header=False)
