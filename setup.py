from setuptools import setup, find_packages

setup(
    name="sar-ship-detection",
    version="0.1.0",
    author="Aaryan Minocha",
    author_email="aaryan2304@gmail.com",
    description="SAR ship detection pipeline for Umbra satellite imagery",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Aaryan2304/sar-ship-detection-pipeline",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.11",
    install_requires=[
        "numpy>=1.24.0",
        "rasterio>=1.3.0",
        "Pillow>=10.0.0",
        "fiftyone>=0.23.0",
        "fiftyone-brain>=0.13.0",
        "attrs>=19.2.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=20.0.0",
            "flake8>=3.8.0",
            "setuptools>=41.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "chip-tiles=pipeline.chip_tiles:main",
            "augment-data=pipeline.augment_data:main",
            "ingest-fiftyone=pipeline.ingest_fiftyone:main",
        ],
    },
    include_package_data=True,
)
