from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="coffee-leaf-defect-detector",
    version="1.0.0",
    author="Sanjana AS",
    author_email="sanjana@example.com",
    description="Deep Learning based Coffee Leaf Disease Detection System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sanjanaaAS/coffee-leaf-defect-detector",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.9",
    install_requires=[
        "tensorflow>=2.13.0",
        "keras>=2.13.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "opencv-python>=4.8.0",
        "pillow>=10.0.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "flask>=2.3.0",
        "pyyaml>=6.0",
    ],
)
