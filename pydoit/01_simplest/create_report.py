"""
Create Report Script
====================

Creates a final report summarizing the processed data.
"""

# Read the processed data
with open("_output/processed.txt") as f:
    content = f.read()

# Write the final report
with open("_output/report.txt", "w") as f:
    f.write("=" * 30 + "\n")
    f.write("      FINAL REPORT\n")
    f.write("=" * 30 + "\n\n")
    f.write(content)
    f.write("\n" + "=" * 30 + "\n")
    f.write("Report generated successfully!\n")
    f.write("=" * 30 + "\n")

print("Created _output/report.txt")
