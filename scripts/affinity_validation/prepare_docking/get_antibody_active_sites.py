import sys

cdr_file = sys.argv[1]
sasa_file = sys.argv[2]

# Read both lists
cdr = set(map(int, open(cdr_file).read().strip().split(',')))
sasa = set(map(int, open(sasa_file).read().strip().split(',')))

# Intersection
active_cdr = sorted(cdr & sasa)

# Print to stdout so you can redirect with >
print(','.join(map(str, active_cdr)))
