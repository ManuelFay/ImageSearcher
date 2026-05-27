import logging
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import yaml

from flask import Flask
from flask import abort, jsonify, request, send_file
from flask_cors import CORS
from PIL import Image

from image_searcher import Search
from image_searcher.api.flask_config import FlaskConfig


class RunFlaskCommand:
    logging.basicConfig(level=logging.INFO)
    web_directory = Path(__file__).parent / "web"

    def __init__(self, config_path: Optional[str] = None, searcher: Optional[Search] = None):
        if config_path is None and searcher is None:
            raise AttributeError("Either the config path or the searcher need to be defined")

        self.searcher = searcher
        self._context_index = None
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

    def _permitted_image_path(self, image_path: str) -> Path:
        """Resolve an image path, restricting browser access to the configured collection."""
        if not image_path:
            abort(400, description="An image path is required.")

        root_path = Path(self.config.image_dir_path).expanduser().resolve()
        resolved_path = Path(image_path).expanduser().resolve()
        try:
            resolved_path.relative_to(root_path)
        except ValueError:
            abort(403, description="This image is outside the configured collection.")

        accepted_formats = (".png", ".jpg", ".jpeg")
        if resolved_path.suffix.lower() not in accepted_formats or not resolved_path.is_file():
            abort(404, description="Image not found.")
        return resolved_path

    def serve_image(self):
        """Return an indexed local image for the browser interface."""
        return send_file(self._permitted_image_path(request.args.get("path") or ""))

    def get_interface_settings(self):
        """Return frontend-safe runtime configuration."""
        return jsonify(include_faces=self.config.include_faces, max_results=self.config.n)

    @staticmethod
    def _formatted_date(timestamp: float) -> str:
        return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()

    def _image_paths(self):
        root_path = Path(self.config.image_dir_path).expanduser().resolve()
        glob_method = root_path.rglob if self.config.traverse else root_path.glob
        return [
            path.resolve() for path in glob_method("*")
            if path.is_file() and path.suffix.lower() in (".png", ".jpg", ".jpeg")
        ]

    def _base_context_entry(self, image_path: Path):
        root_path = Path(self.config.image_dir_path).expanduser().resolve()
        relative_path = image_path.relative_to(root_path)
        conversation_key = relative_path.parts[0] if len(relative_path.parts) > 1 else root_path.name
        timestamp = image_path.stat().st_mtime
        return {
            "image_path": str(image_path),
            "captured_at": self._formatted_date(timestamp),
            "timestamp": timestamp,
            "date_source": "File modified date",
            "conversation_key": conversation_key,
            "conversation": conversation_key,
        }

    def _apply_messenger_metadata(self, index):
        root_path = Path(self.config.image_dir_path).expanduser().resolve()
        if root_path.name != "inbox":
            return

        entries_by_thread = {}
        for entry in index.values():
            entries_by_thread.setdefault(entry["conversation_key"], []).append(entry)

        for message_path in root_path.glob("*/message_*.json"):
            try:
                with message_path.open("r", encoding="utf-8") as message_file:
                    thread = json.load(message_file)
            except (OSError, json.JSONDecodeError):
                continue

            thread_key = message_path.parent.name
            thread_title = thread.get("title") or thread_key
            for entry in entries_by_thread.get(thread_key, []):
                entry["conversation"] = thread_title

            for message in thread.get("messages", []):
                timestamp_ms = message.get("timestamp_ms")
                if timestamp_ms is None:
                    continue
                for photo in message.get("photos", []):
                    uri = photo.get("uri") or ""
                    marker = "inbox/"
                    relative_uri = uri.split(marker, 1)[-1] if marker in uri else uri
                    image_path = str((root_path / relative_uri).resolve())
                    if image_path in index:
                        timestamp = timestamp_ms / 1000
                        index[image_path].update({
                            "captured_at": self._formatted_date(timestamp),
                            "timestamp": timestamp,
                            "date_source": "Messenger message date",
                            "sender": message.get("sender_name"),
                        })

    @staticmethod
    def _exif_date(image_path: Path):
        try:
            with Image.open(image_path) as image:
                exif = image.getexif()
                raw_date = exif.get(36867) or exif.get(36868) or exif.get(306)
            if raw_date:
                return datetime.strptime(raw_date, "%Y:%m:%d %H:%M:%S")
        except (OSError, ValueError):
            return None
        return None

    def _apply_exif_metadata(self, index):
        for entry in index.values():
            if entry["date_source"] != "File modified date":
                continue
            captured_at = self._exif_date(Path(entry["image_path"]))
            if captured_at:
                entry.update({
                    "captured_at": captured_at.isoformat(),
                    "timestamp": captured_at.timestamp(),
                    "date_source": "EXIF capture date",
                })

    def _build_context_index(self):
        index = {str(path): self._base_context_entry(path) for path in self._image_paths()}
        self._apply_messenger_metadata(index)
        self._apply_exif_metadata(index)
        return index

    def _image_metadata(self, image_path: Path, entry):
        metadata = dict(entry)
        try:
            with Image.open(image_path) as image:
                metadata["width"], metadata["height"] = image.size
        except OSError:
            pass
        metadata["size_bytes"] = image_path.stat().st_size
        return metadata

    @staticmethod
    def _related_image(entry, selected_timestamp):
        return {
            "image_path": entry["image_path"],
            "captured_at": entry["captured_at"],
            "date_source": entry["date_source"],
            "time_difference_seconds": abs(entry["timestamp"] - selected_timestamp),
        }

    def get_image_context(self):
        """Return metadata and contextual navigation options for a selected image."""
        image_path = self._permitted_image_path(request.args.get("path") or "")
        if self._context_index is None:
            self._context_index = self._build_context_index()
        selected = self._context_index.get(str(image_path), self._base_context_entry(image_path))
        metadata = self._image_metadata(image_path, selected)
        selected_timestamp = metadata["timestamp"]
        limit = min(self.config.n, 16)
        candidates = [entry for path, entry in self._context_index.items() if path != str(image_path)]
        same_conversation = [
            entry for entry in candidates if entry["conversation_key"] == metadata["conversation_key"]
        ]
        same_conversation.sort(key=lambda entry: abs(entry["timestamp"] - selected_timestamp))
        candidates.sort(key=lambda entry: abs(entry["timestamp"] - selected_timestamp))
        return jsonify(
            metadata={key: value for key, value in metadata.items() if key != "timestamp"},
            same_conversation=[
                self._related_image(entry, selected_timestamp) for entry in same_conversation[:limit]
            ],
            nearby_time=[self._related_image(entry, selected_timestamp) for entry in candidates[:limit]],
        )

    def run(self, start=True):
        app = Flask(__name__, static_folder=str(self.web_directory), static_url_path="/assets")
        self.searcher = Search(image_dir_path=self.config.image_dir_path,
                               traverse=self.config.traverse,
                               save_path=self.config.save_path,
                               reindex=self.config.reindex,
                               include_faces=self.config.include_faces)
        CORS(app)
        app.add_url_rule("/", "home", lambda: app.send_static_file("index.html"), methods=["GET"])
        app.add_url_rule("/media", "serve_image", self.serve_image, methods=["GET"])
        app.add_url_rule("/settings", "settings", self.get_interface_settings, methods=["GET"])
        app.add_url_rule("/image_context", "image_context", self.get_image_context, methods=["GET"])
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
