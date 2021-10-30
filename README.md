# ImageSearcher
### Leveraging CLIP to perform image search on personal pictures

This repository implements an Image Search engine on local photos powered by the CLIP model.
It is surprisingly precise and is able to find images given complex queries. For more information, refer to
the Medium blogpost [here](https://medium.com/@manuelfaysse/building-a-powerful-image-search-engine-for-your-pictures-using-deep-learning-16d06df10385?source=friends_link&sk=ca5130cb63a1fcb3a3e3f54ff494e56b).

The added functionality of classifying pictures depending on the persons portrayed is implemented 
with the `face_recognition` library. Several filters are also available, enabling you to find your 
group pictures, screenshots, etc...

## Setup

In a new Python 3.8+ virtual environment run:
```bash
pip install -r dev_requirements.txt
pip install face_recognition  # Optional
```

Troubleshooting: If problems are encountered building wheels for dlib during the face_recognition installation, make sure to install the `python3.8-dev`
package (respectively `python3.x-dev`) and recreate the virtual environment from scratch with the aforementionned command 
once it is installed.

## Usage
Currently, the usage is as follows. It computes the embeddings of all images one by one, and stores them in 
a picked dictionary for further reference. To compute and store information about the persons in 
the picture, enable the `include_faces` flag (note that it makes the indexing process up to 10x slower).

```python
from image_searcher import Search

searcher = Search(image_dir_path="/home/manu/perso/ImageSearcher/data/", traverse=True, include_faces=False)
ranked_images = searcher.rank_images("A photo of a bird.", n=5)

# Display best images
from PIL import Image

for image in ranked_images:
    Image.open(image.image_path).convert('RGB').show()
```

### Using tags in the query
Adding tags at the end of the query (example: `A bird singing #photo`) will filter the search based on the tag list.
Supported tags for the moment are:
  - \#{category}: Amongst "screenshot", "drawing", "photo", "schema", "selfie"
  - \#groups: Group pictures (more than 5 people)

To come is support for:
  
  - \#dates: Filtering based on the time period


### Running through the API for efficient use

After having indexed the images of interest, a Flask API can be used to load models once and then search efficiently.

#### Specify a Config YAML file:
  
```yaml
image_dir_path: /home/manu/Downloads/facebook_logs/messages/inbox/
save_path: /home/manu/
traverse: true
include_faces: true
reindex: false
n: 42

port:
host:
debug:
threaded:
```
#### Start a server:
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
    --timeout 30
```

Note: Adapt the timeout parameter (in seconds) if a lot of new images are being indexed/


#### Query it:

- By opening in a browser the webpage with the demo search engine `search.html`.

- Through the API endpoint online: `http://127.0.0.1:5000/get_best_images?q=a+photo+of+a+bird`

- In Python:
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

This repo is a work in progress that has recently been started. As is, it computes about 10 images per second during the initial indexing phase, then is almost instantaneous during the querying phase.

Feature requests and contributions are welcomed. Improvements to the Search Web interface would also
be greatly appreciated !

## Todo list

Simplify and robustify the Search class instanciation:
- Check indexation arguments are compatible with pre-loaded file
- Store indexation arguments in pre-loaded file and give option to index new pictures with these options
- Add the option to index for faces on previously CLIP indexed images

Speed:
- Parallel indexation / dynamic batching based on image size
- Data loader before indexation
- Optimized vector computation with optimized engine (FAISS)

Features:
- Image auto-tagging (screenshot, drawing, photo, nature, group picture, selfie, etc)
- Image deduplication (perceptual hashing)

Embedding files:
- Integrate with local version control (git-lfs ?)

Frontend:
- Overall UX and design changes
- Enable Image upload

Deployment:
- Dockerize and orchestrate containers (image uploader, storage, indexation pipeline, inference)