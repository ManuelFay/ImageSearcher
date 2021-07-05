from setuptools import setup, find_packages

setup(
    name="Image-Search",
    version="DEV",
    description="Image Search module",
    author="Manuel Faysse",
    author_email='manuel.fay@gmail.com',
    packages=find_packages(include=["image_searcher", "image_searcher.*"]),
    install_requires=[
        # "jax",
        # "flax",
        "torch",
        "ftfy",
        "transformers",
        "Pillow"
    ],
    python_requires=">=3.7,<4.0",
)
