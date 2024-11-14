from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

class HandcraftedDescriptors(): 
    
    def __init__(self, method='ECFP', radius=2, fpSize=2048):
        super().__init__()
        assert method == 'ECFP', f'{method} is not a supported method for handcrafted descriptors.'
        self.radius = radius
        self.fpSize = fpSize

    def encode(self, smiles_list):
        fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=self.radius, fpSize=self.fpSize)
        fp = [fpgen.GetFingerprint(Chem.MolFromSmiles(s)).ToList() for s in smiles_list]
        return fp

