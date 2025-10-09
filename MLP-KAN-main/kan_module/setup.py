from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="kan-module",
    version="1.0.0",
    author="KAN Module Team",
    author_email="",
    description="Kolmogorov-Arnold Networks (KAN) implementation with PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/kan-module",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.13.1",
        "torchvision>=0.8.1",
    ],
    extras_require={
        "vision": ["timm>=0.3.2"],
        "dev": ["numpy>=1.21.0", "matplotlib>=3.5.0"],
        "all": ["timm>=0.3.2", "numpy>=1.21.0", "matplotlib>=3.5.0"],
    },
    keywords="kan, kolmogorov-arnold, neural-networks, pytorch, deep-learning",
    project_urls={
        "Bug Reports": "https://github.com/your-username/kan-module/issues",
        "Source": "https://github.com/your-username/kan-module",
    },
)