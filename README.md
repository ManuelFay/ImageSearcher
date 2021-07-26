# ImageSearcher
### Leveraging CLIP to perform image search on personal pictures

This repository implements an Image Search engine on local photos powered by the CLIP model.
It is surprisingly precise and is able to find images given complex queries. For more information, refer to
the Medium blogpost [here](https://medium.com/@manuelfaysse/building-a-powerful-image-search-engine-for-your-pictures-using-deep-learning-16d06df10385?source=friends_link&sk=ca5130cb63a1fcb3a3e3f54ff494e56b).

## Setup

In a new Python 3.8+ virtual environment run:
```bash
pip install -r dev_requirements.txt
```
## Usage
Currently, the usage is as follows. It computes the embeddings of all images one by one, and stores them in 
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

After having indexed the images of interest, a Flask API can be used to load models once and then search efficiently.

- Specify a Config YAML file:
  
```yaml
image_dir_path: /home/manu/Downloads/facebook_logs/messages/inbox/
save_path: /home/manu/
traverse: false
n: 42

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
    --bind 0.0.0.0:${GUNICORN_PORT:-5000} \
    --worker-tmp-dir /dev/shm \
    --workers=${GUNICORN_WORKERS:-2} \
    --threads=${GUNICORN_THREADS:-4} \
    --worker-class=gthread \
    --log-level=info \
    --log-file '-' \
    --timeout 20
```
- Query it:

By opening in a browser the webpage with the demo search engine `search.html`.

Through the API endpoint online: `http://127.0.0.1:5000/get_best_images?q=a+photo+of+a+bird`

In Python:
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

This repo is a work in progress that has recently been started. Support for batching computations and image loading,
using FAISS or other optimized libraries for vector computation, as well as image deduplication and adding additional info to the 
API answer is ongoing. As is, it computes about 10 images per second during the initial indexing phase, then is almost instantaneous during the querying phase.

Feature requests and contributions are welcomed. Improvements to the Search Web interface would also
be greatly appreciated !