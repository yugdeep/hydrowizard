from setuptools import setup, find_packages

# Read the contents of your README file
with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="hydrowizard",
    version="0.1.7",  # This should match the version in pyproject.toml
    description="A tool for managing water resources",
    long_description=long_description,
    long_description_content_type='text/markdown',
    author="Yugdeep Bangar",
    author_email="yugdeep@hydrowizard.ai",
    url="https://github.com/yugdeep/hydrowizard",
    packages=find_packages(exclude=('tests',)),
    install_requires=[
        "graphviz>=0.20.3,<0.21.0",
        "matplotlib>=3.9.1,<4.0.0",
        "networkx>=3.3,<4.0",
        "numpy>=2.0.1,<3.0.0",
        "openpyxl>=3.1.5,<4.0.0",
        "pandas>=2.2.2,<3.0.0",
        "psycopg2-binary>=2.9.9,<3.0.0",
        "pymoo>=0.6.1.3,<0.7.0",
        "pytz>=2024.1",
        "pyyaml>=6.0.1,<7.0.0",
        "sqlalchemy>=2.0.31,<3.0.0",
        "tqdm>=4.66.4,<5.0.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
    entry_points={
        'console_scripts': [
            'hw-simulation=hydrowizard.scripts.run_simulation:main_entry',
            'hw-optimization=hydrowizard.scripts.run_optimization:main_entry',
        ],
    },
)