# WIP: ImageSearcher
### Leveraging CLIP to perform image search on personal pictures

This repository aims to implement an Image Search engine powered by the CLIP model.

## Setup

In a new Python 3.8+ virtual environment run:
```bash
pip install -r dev_requirements.txt
```
## Usage
 Currently, the naive usage is as follows. Note that it computes the embeddings of all images for each query,
and crashes when RAM runs outs.
```python
from image_searcher.interfaces.search_interface import NaiveSearch

searcher = NaiveSearch(image_dir_path="/home/manu/perso/ImageSearcher/data/")
ranked_images = searcher.rank_images("A photo of a bird.")
print(ranked_images)
```
## Tests

Run the tests with 

```bash
python -m unittest
```

## Contributing

This repo is largely a work in progress that has only recently been started. Support for pre-computing 
the image embeddings, batching computations and image loading, and using FAISS or other optimized libraries for vector computation is ongoing.