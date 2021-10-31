import logging
from typing import Optional
import yaml

from flask import Flask
from flask import request, jsonify
from flask_cors import CORS

from image_searcher import Search
from image_searcher.api.flask_config import FlaskConfig


class RunFlaskCommand:
    logging.basicConfig(level=logging.INFO)

    def __init__(self, config_path: Optional[str] = None, searcher: Optional[Search] = None):
        if config_path is None and searcher is None:
            raise AttributeError("Either the config path or the searcher need to be defined")

        self.searcher = searcher
        if config_path:
            self.config = FlaskConfig(**yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader))
        else:
            self.config = FlaskConfig(image_dir_path=self.searcher.loader.image_dir_path,
                                      traverse=self.searcher.loader.traverse,
                                      save_path=self.searcher.stored_embeddings.save_path,
                                      reindex=False,
                                      include_faces=len(self.searcher.stored_embeddings.face_embedding_paths) > 0)

    def get_best_images(self):
        """Routine that runs the QA inference pipeline.
        """
        user_query = request.args.get("q") or ""

        logging.info(f"User query: {user_query}")
        result = self.searcher.rank_images(user_query, n=self.config.n)
        # logging.info(result)
        return jsonify(results=result)

    def get_closest_faces(self):
        """Routine that runs the QA inference pipeline.
        """
        image_query = request.args.get("q") or ""

        logging.info(f"Image query: {image_query}")
        result = self.searcher.rank_images_by_faces(image_query, n=self.config.n)
        # logging.info(result)
        return jsonify(results=result)

    def run(self, start=True):
        app = Flask(__name__, static_folder=None)
        self.searcher = Search(image_dir_path=self.config.image_dir_path,
                               traverse=self.config.traverse,
                               save_path=self.config.save_path,
                               reindex=self.config.reindex,
                               include_faces=self.config.include_faces)
        CORS(app)
        app.add_url_rule("/get_best_images", "get_best_images", self.get_best_images, methods=["GET"])
        app.add_url_rule("/get_closest_faces", "get_closest_faces", self.get_closest_faces, methods=["GET"])

        if start:
            app.run(port=self.config.port,
                    host=self.config.host,
                    debug=self.config.debug,
                    threaded=self.config.threaded)
        return app


def run(**kwargs):
    command = RunFlaskCommand(**kwargs)
    command.run()
