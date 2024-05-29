import os

header_template = """
This file is part of the
  Garpar Project (https://github.com/quatrope/garpar).
Copyright (c) 2021, 2022, Nadia Luczywo, Juan Cabral and QuatroPe
License: MIT
  Full Text: https://github.com/quatrope/garpar/blob/master/LICENSE
"""

# Lista de archivos con encabezados incorrectos
files_to_update = [
    'garpar/utils/mabc.py',
    'garpar/utils/scalers.py',
    'garpar/utils/entropy.py',
    'garpar/core/utilities_acc.py',
    'garpar/core/risk_acc.py',
    'garpar/core/__init__.py',
    'garpar/core/_mixins.py',
    'garpar/core/div_acc.py',
    'garpar/core/covcorr_acc.py',
    'garpar/core/ereturns_acc.py',
    'garpar/datasets/multisector.py',
    'garpar/datasets/data/__init__.py',
]

for file_path in files_to_update:
    with open(file_path, 'r') as file:
        content = file.readlines()
    
    # Remove old header
    content = [line for line in content if not line.startswith('"""')]
    
    # Add new header
    new_content = [header_template] + content
    
    with open(file_path, 'w') as file:
        file.writelines(new_content)

print("Headers updated successfully.")