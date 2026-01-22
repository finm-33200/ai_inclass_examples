"""
Demonstrates storing WRDS credentials using environment variables.

Before running:
    cp .env.example .env
    # Edit .env with your WRDS username
    python demo_wrds_config.py
"""
from decouple import config

wrds_username = config("WRDS_USERNAME")
print(f"WRDS_USERNAME = {wrds_username}")

# In real usage, you would connect to WRDS like this:
#
# import wrds
# db = wrds.Connection(wrds_username=wrds_username)
#
# The password is stored in ~/.pgpass (not in .env)
# See: https://finm-32900.github.io/notebooks/_01_wrds_python_package.html
