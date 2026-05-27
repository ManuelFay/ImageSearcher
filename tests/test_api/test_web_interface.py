import json
import os
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from image_searcher.api.run_flask_command import RunFlaskCommand


class FakeSearcher:
    def __init__(self, **_kwargs):
        pass

    def rank_images(self, _query, n=10):
        return []

    def rank_images_by_faces(self, _path, n=10):
        return []


class TestWebInterface(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.collection_path = os.path.join(self.tempdir.name, "inbox")
        thread_path = os.path.join(self.collection_path, "chat_one", "photos")
        other_thread_path = os.path.join(self.collection_path, "chat_two", "photos")
        os.makedirs(thread_path)
        os.makedirs(other_thread_path)
        self.image_path = os.path.join(thread_path, "test.jpg")
        self.neighbor_path = os.path.join(thread_path, "neighbor.jpg")
        self.near_time_path = os.path.join(other_thread_path, "near-time.jpg")
        for image_path in (self.image_path, self.neighbor_path, self.near_time_path):
            Image.new("RGB", (12, 8), color="white").save(image_path)

        messages = {
            "title": "Test Conversation",
            "messages": [
                {
                    "timestamp_ms": 1700000000000,
                    "sender_name": "Alice",
                    "photos": [{"uri": "messages/inbox/chat_one/photos/test.jpg"}],
                },
                {
                    "timestamp_ms": 1700000060000,
                    "sender_name": "Bob",
                    "photos": [{"uri": "messages/inbox/chat_one/photos/neighbor.jpg"}],
                },
            ],
        }
        with open(os.path.join(self.collection_path, "chat_one", "message_1.json"), "w") as message_file:
            json.dump(messages, message_file)
        other_messages = {
            "title": "Other Conversation",
            "messages": [{
                "timestamp_ms": 1700000030000,
                "sender_name": "Charlie",
                "photos": [{"uri": "messages/inbox/chat_two/photos/near-time.jpg"}],
            }],
        }
        with open(os.path.join(self.collection_path, "chat_two", "message_1.json"), "w") as message_file:
            json.dump(other_messages, message_file)

        config = type("Config", (), {
            "image_dir_path": self.collection_path,
            "save_path": self.collection_path,
            "traverse": True,
            "reindex": False,
            "include_faces": True,
            "n": 10,
            "port": None,
            "host": None,
            "debug": None,
            "threaded": True,
        })()
        self.command = RunFlaskCommand.__new__(RunFlaskCommand)
        self.command.config = config
        self.command._context_index = None
        with patch("image_searcher.api.run_flask_command.Search", FakeSearcher):
            self.client = self.command.run(start=False).test_client()

    def tearDown(self):
        self.tempdir.cleanup()

    def test_home_serves_interface(self):
        response = self.client.get("/")
        self.addCleanup(response.close)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Find the moment you remember.", response.data)

    def test_media_serves_configured_image(self):
        response = self.client.get("/media", query_string={"path": self.image_path})
        self.addCleanup(response.close)
        self.assertEqual(response.status_code, 200)

    def test_media_rejects_image_outside_collection(self):
        response = self.client.get("/media", query_string={"path": "/tmp/not-in-library.jpg"})
        self.assertEqual(response.status_code, 403)

    def test_settings_exposes_face_capability(self):
        response = self.client.get("/settings")
        self.assertEqual(response.json["include_faces"], True)

    def test_context_uses_message_metadata_and_related_photos(self):
        response = self.client.get("/image_context", query_string={"path": self.image_path})
        payload = response.json
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["metadata"]["date_source"], "Messenger message date")
        self.assertEqual(payload["metadata"]["conversation"], "Test Conversation")
        self.assertEqual(payload["metadata"]["sender"], "Alice")
        self.assertEqual(payload["metadata"]["width"], 12)
        self.assertEqual(payload["same_conversation"][0]["image_path"], self.neighbor_path)
        self.assertEqual(payload["nearby_time"][0]["image_path"], self.near_time_path)
