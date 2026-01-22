"""
Explore the PATH environment variable.

Run:
    python explore_path.py
"""
import os

path = os.environ.get("PATH", "")
dirs = path.split(os.pathsep)

print("Your PATH contains these directories:\n")
for i, d in enumerate(dirs, 1):
    print(f"{i:2}. {d}")

print(f"\nTotal: {len(dirs)} directories")
print(f"PATH separator on this system: {repr(os.pathsep)}")
