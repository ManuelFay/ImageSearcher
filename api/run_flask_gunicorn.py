from flask import Flask
from image_searcher.api.run_flask_command import RunFlaskCommand


def create_app(config_path: str) -> Flask:
    """Create the Flask app object and return it (without starting the app).
    Entry point for gunicorn
    """

    command = RunFlaskCommand(config_path=config_path)
    return command.run(start=False)
