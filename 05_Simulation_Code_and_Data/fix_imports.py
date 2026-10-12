import sys

# Read the file
with open('run_batch_integrated.py', 'r') as f:
    lines = f.readlines()

# Check if import sys exists
has_sys_import = any('import sys' in line for line in lines)

if not has_sys_import:
    # Add import sys at the beginning
    new_lines = ['import sys\n'] + lines
    with open('run_batch_integrated.py', 'w') as f:
        f.writelines(new_lines)
    print("Added import sys")
else:
    print("import sys already exists")

# Also check for other missing imports
required_imports = ['import sys', 'import os', 'import numpy', 'import pandas']
for imp in required_imports:
    if not any(imp in line for line in lines):
        print(f"Missing: {imp}")
