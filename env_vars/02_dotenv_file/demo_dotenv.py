"""
Demonstrates loading environment variables from a .env file.

Before running:
    cp .env.example .env
    python demo_dotenv.py
"""
from dotenv import load_dotenv
import os

# Load variables from .env into os.environ
load_dotenv()

data_dir = os.getenv("DATA_DIR")
output_dir = os.getenv("OUTPUT_DIR")

print(f"DATA_DIR = {data_dir}")
print(f"OUTPUT_DIR = {output_dir}")
