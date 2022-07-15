from setuptools import setup, find_packages
from pathlib import Path

extras = {
    "face_recognition": ["face_recognition==1.3.0"],
    "server": ["flask>=2.0.0, <3.0.0", "flask_cors"]
}

setup(
    name="image-searcher",
    version="v0.0.3",
    description="Image Searcher based on semantic query understanding for your own pictures.",
    long_description=(Path(__file__).parent / "README.md").read_text(),
    long_description_content_type='text/markdown',
    author="Manuel Faysse",
    author_email='manuel.fay@gmail.com',
    download_url="https://github.com/ManuelFay/ImageSearcher/archive/refs/tags/v0.0.3.tar.gz",
    url="https://github.com/ManuelFay/ImageSearcher",
    keywords=['search engine', 'image', 'image search', 'CLIP'],
    packages=find_packages(include=["image_searcher", "image_searcher.*"]),
    install_requires=[
        "torch>=1.7.0,<1.13.0",
        "numpy>=1.20.0,<2.0.0",
        "ftfy>=6.0.0, <7.0.0",
        "transformers>=4.9.0,<5.0.0",
        "Pillow>=8.3.0,<9.0.0",
    ],
    extras_require=extras,
    python_requires=">=3.7,<4.0",
)
