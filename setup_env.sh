#!/bin/bash
# Filename: setup_poetry_environment.sh

# Deactivate the virtual environment if it's active
if [[ "$VIRTUAL_ENV" != "" ]]; then
    deactivate
fi

# Remove existing .venv directory
rm -rf .venv

# Create a new virtual environment
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install Poetry 1.8.3
pip install poetry==1.8.3

# Run poetry install with dev and docs extras
poetry install --with dev,docs

echo "Setup complete!"