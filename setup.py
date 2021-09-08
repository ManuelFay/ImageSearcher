from setuptools import setup, find_packages

extras = {
    "face_recognition": ["face_recognition"]
}

setup(
    name="Image-Search",
    version="DEV",
    description="Image Search module",
    author="Manuel Faysse",
    author_email='manuel.fay@gmail.com',
    packages=find_packages(include=["image_searcher", "image_searcher.*"]),
    install_requires=[
        "torch",
        "ftfy",
        "transformers",
        "Pillow",
    ],
    extras_require=extras,
    python_requires=">=3.7,<4.0",
)
