import os


class ImageLoader:
    def __init__(self, image_dir_path: str, traverse=False):
        self.image_dir_path = image_dir_path
        self.traverse = traverse
        self.accepted_formats = (".png", ".jpg", ".jpeg")

    def search_tree(self):
        if self.traverse:
            image_files = [os.path.join(root, file) for root, dirs, files in os.walk(self.image_dir_path) for file in files
                           if file.endswith(self.accepted_formats)]
        else:
            image_files = [os.path.join(self.image_dir_path, file) for file in os.listdir(self.image_dir_path)
                           if file.endswith(self.accepted_formats)]
        return image_files
