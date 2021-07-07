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

searcher = Search(image_dir_path="/home/manu/perso/ImageSearcher/data/", traverse=True)
ranked_images = searcher.rank_images("A photo of a bird.", n=5)

# Display best images
from PIL import Image

for image in ranked_images:
    Image.open(image.image_path).convert('RGB').show()
```

### Running through the API for efficient use

The Flask API can be used to load models once and then search efficiently:

- Specify a Config YAML file:
  
```yaml
image_dir_path: /home/manu/Downloads/facebook_logs/messages/inbox/
traverse: false

port:
host:
debug:
threaded:
```
- Start a server:
```python
from api.run_flask_command import RunFlaskCommand

command = RunFlaskCommand(config_path="/home/manu/perso/ImageSearcher/api/api_config.yml")
command.run()
```

A gunicorn process can also be launched locally with:

```bash
gunicorn "api.run_flask_gunicorn:create_app('/home/manu/perso/ImageSearcher/api/api_config.yml')" \
    --name image_searcher \
    --bind 0.0.0.0:${GUNICORN_PORT:-8000} \
    --worker-tmp-dir /dev/shm \
    --workers=${GUNICORN_WORKERS:-2} \
    --threads=${GUNICORN_THREADS:-4} \
    --worker-class=gthread \
    --log-level=info \
    --log-file '-' \
    --timeout 20
```
- Query it:

Through the API endpoint online: http://127.0.0.1:5000/get_best_images?q=a+photo+of+a+bird

Or in Python:
```python
import requests
import json
import urllib.parse

query = "a photo of a bird"
r = requests.get(f"http://127.0.0.1:5000/get_best_images?q={urllib.parse.quote(query)}")
print(json.loads(r.content)["results"])
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
about 10 images per second during the initial indexing phase, then is almost instantaneous during the querying 
phase.

Feature requests and contributions are welcomed.