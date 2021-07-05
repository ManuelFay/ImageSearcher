# WIP: ImageSearcher
### Leveraging CLIP to perform image search on personal pictures

This repository implements an Image Search engine on local photos powered by the CLIP model.
From initial tests, it looks very powerful and is able to find images given complex queries.

## Setup

In a new Python 3.8+ virtual environment run:
```bash
pip install -r dev_requirements.txt
```
## Usage
Currently, the non-optimized usage is as follows. It computes the embeddings of all images one by one, and stores them in 
a picked dictionary for further reference.

```python
from image_searcher import Search

searcher = Search(image_dir_path="/home/manu/perso/ImageSearcher/data/")
ranked_images = searcher.rank_images("A photo of a bird.")

# Display best images
from PIL import Image

for image, conf in ranked_images:
    Image.open(image).convert('RGB').show()
```

For testing purposes, the naive usage is as follows. Note that it computes the embeddings of all images for each query,
and crashes when RAM runs outs.

```python
from image_searcher.search.naive_search import NaiveSearch

searcher = NaiveSearch(image_dir_path="/home/manu/perso/ImageSearcher/data/")
ranked_images = searcher.rank_images("A photo of a bird.")
print(ranked_images)
```

### Tips

Using this tool with vacation photos, or Messenger and Whatsapp photo archives leads to rediscovering 
old photos and is amazing at locating long lost ones.

## Tests

Run the tests with 

```bash
python -m unittest
```

and lint with:

```bash
pylint image_searcher
```

## Contributing

This repo is a work in progress that has only recently been started. Support for batching computations and image loading,
and using FAISS or other optimized libraries for vector computation is ongoing. As is, it computes
about 5 images per second during the initial indexing phase, then is almost instantaneous during the querying 
phase.

Feature requests and contributions are welcomed.