from typing import List, Generator
import os

from PIL import Image


class ImageLoader:
    def __init__(self, image_dir_path: str, traverse=False):
        self.image_dir_path = image_dir_path
        self.traverse = traverse
        self.accepted_formats = (".png", ".jpg", ".jpeg")
        self.batch_size = 3

    def search_tree(self):
        if self.traverse:
            image_files = [os.path.join(root, file) for root, dirs, files in os.walk(self.image_dir_path) for file in
                           files
                           if file.endswith(self.accepted_formats)]
        else:
            image_files = [os.path.join(self.image_dir_path, file) for file in os.listdir(self.image_dir_path)
                           if file.endswith(self.accepted_formats)]
        return image_files

    def open_images(self, image_paths: List[str]) -> Generator:
        for idx in range(0, len(image_paths), self.batch_size):
            yield [Image.open(file) for file in image_paths[idx:min(idx+self.batch_size, len(image_paths))]]
