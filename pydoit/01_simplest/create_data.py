"""
Create Data Script
==================

Creates initial data file with some sample content.
"""

from pathlib import Path

# Create output directory if it doesn't exist
Path("_output").mkdir(exist_ok=True)

# Write sample data
with open("_output/data.txt", "w") as f:
    f.write("Apple\n")
    f.write("Banana\n")
    f.write("Cherry\n")

print("Created _output/data.txt")
