import logging
import yaml

from flask import Flask
from flask import request, jsonify
from flask_cors import CORS

from image_searcher import Search
from api.flask_config import FlaskConfig


class RunFlaskCommand:
    logging.basicConfig(level=logging.INFO)

    def __init__(self, config_path: str):
        self.searcher = None
        self.config = FlaskConfig(**yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader))

    def get_best_images(self):
        """Routine that runs the QA inference pipeline.
        """
        user_query = request.args.get("q") or ""

        logging.info(f"User query: {user_query}")
        result = self.searcher.rank_images(user_query, n=self.config.n)
        # logging.info(result)
        return jsonify(results=result)

    def run(self, start=True):
        app = Flask(__name__, static_folder=None)
        self.searcher = Search(image_dir_path=self.config.image_dir_path,
                               traverse=self.config.traverse,
                               save_path=self.config.save_path)
        CORS(app)
        app.add_url_rule("/get_best_images", "get_best_images", self.get_best_images, methods=["GET"])

        if start:
            app.run(port=self.config.port,
                    host=self.config.host,
                    debug=self.config.debug,
                    threaded=self.config.threaded)
        return app


if __name__ == "__main__":
    command = RunFlaskCommand(config_path="/home/manu/perso/ImageSearcher/api/api_config.yml")
    command.run()
