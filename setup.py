"""
MRI-GUARDIAN: Physics-Guided Generative MRI Reconstruction and Hallucination Auditor
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mri-guardian",
    version="1.0.0",
    author="Mukund Thiru",
    author_email="",
    description="Physics-Guided MRI Reconstruction and Hallucination Auditor",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/contactmukundthiru-cyber/MRI-GUARDIAN",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "h5py>=3.8.0",
        "scikit-image>=0.20.0",
        "matplotlib>=3.7.0",
        "PyYAML>=6.0",
        "tqdm>=4.65.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.2.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
        "wandb": ["wandb>=0.14.0"],
    },
    entry_points={
        "console_scripts": [
            "mri-guardian-train=scripts.train_guardian:main",
            "mri-guardian-eval=scripts.evaluate:main",
        ],
    },
)
