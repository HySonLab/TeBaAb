import sys
import re
import shutil

if len(sys.argv) != 2:
    print("Usage: python patch_run_cns.py <file_path>")
    sys.exit(1)

file_path = sys.argv[1]

# --- Make a backup before modification ---
shutil.copy(file_path, file_path + ".bak")

# --- Old and new content blocks ---
old_block = r"""
\{===================== Number of structures to dock =======================\}
\{\* Setting for the rigid-body \(it0\) and semi-flexible refiment \(it1\) \*\}

\{\* number of structures for rigid body docking \*\}
\{===>\} structures_0=\d+;
\s+keepstruct_0=&structures_0;
\{\* number of structures for refinement \*\}
\{===>\} structures_1=\d+;
\s+keepstruct_1=&structures_1;
\s+keepstruct_2=&structures_1;
\{\* number of structures to be analysed\*\}
\{===>\} anastruc_1=\d+;
\s+anastruc_0=&anastruc_1;
\s+anastruc_2=&anastruc_1;
"""

new_block = """{===================== Number of structures to dock =======================}
{* Setting for the rigid-body (it0) and semi-flexible refiment (it1) *}

{* number of structures for rigid body docking *}
{===>} structures_0=200;
       keepstruct_0=&structures_0;
{* number of structures for refinement *}
{===>} structures_1=40;
       keepstruct_1=&structures_1;
       keepstruct_2=&structures_1;
{* number of structures to be analysed*}
{===>} anastruc_1=40;
       anastruc_0=&anastruc_1;
       anastruc_2=&anastruc_1;
"""

# --- Read, replace, and write back ---
with open(file_path, "r") as f:
    content = f.read()

new_content = re.sub(old_block, new_block, content, flags=re.MULTILINE)

if content == new_content:
    print("⚠️ No matching block found or already updated.")
else:
    with open(file_path, "w") as f:
        f.write(new_content)
    print(f"✅ Replacement complete! Backup saved as {file_path}.bak")
