"""
Process Data Script
===================

Processes the data - numbers each line and adds a count.
"""

# Read the data
with open("_output/data.txt") as f:
    lines = f.readlines()

# Write processed data
with open("_output/processed.txt", "w") as f:
    f.write(f"Total items: {len(lines)}\n")
    f.write("-" * 20 + "\n")
    for i, line in enumerate(lines, 1):
        f.write(f"{i}. {line}")

print("Created _output/processed.txt")
