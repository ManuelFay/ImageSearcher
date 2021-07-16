# pylint:disable=no-member
from typing import List
import logging

import torch
from tqdm import tqdm

from image_searcher.interfaces.image_loader import ImageLoader
from image_searcher.interfaces.result_interface import RankedImage
from image_searcher.interfaces.stored_embeddings import StoredEmbeddings
from image_searcher.embedders.clip_embedder import ClipEmbedder


class Search:
    def __init__(self, image_dir_path: str, traverse: bool = False, save_path: str = None):
        self.loader = ImageLoader(image_dir_path=image_dir_path, traverse=traverse)

        logging.info("Loading CLIP Embedder")
        self.embedder = ClipEmbedder()

        logging.info("Loading pre-computed embeddings")
        self.stored_embeddings = StoredEmbeddings(save_path=save_path if save_path else image_dir_path)
        logging.info(f"{len(self.stored_embeddings.get_image_paths())} files are indexed.")

        logging.info(f"Re-indexing the image files in {image_dir_path}")
        self.reindex()

        logging.info(f"Excluding files from search if not in {image_dir_path}")
        self.stored_embeddings.exclude_extra_files(filter_path=image_dir_path)

        logging.info(f"Setup over, Searcher is ready to be queried")

    def reindex(self):
        waiting_list = set(self.loader.search_tree()) - set(self.stored_embeddings.get_image_paths())
        if not waiting_list:
            return

        for idx, image_path in enumerate(tqdm(waiting_list)):
            try:
                image = [self.loader.open_image(image_path)]
                self.stored_embeddings.add_embedding(image_path, self.embedder.embed_images(image))
                if idx % 1000 == 0:
                    self.stored_embeddings.update_file()

            except Exception as exception:
                logging.warning(f"Image {image_path} has failed to process - adding it to fail list.")
                self.stored_embeddings.add_embedding(image_path, torch.zeros((1, 512)))
                logging.warning(exception)

        self.stored_embeddings.update_file()

    def rank_images(self, query: str, n: int = 10) -> List[RankedImage]:
        assert isinstance(query, str)
        text_embeds = self.embedder.embed_text(query)
        image_embeds = self.stored_embeddings.get_embedding_tensor()
        image_paths = self.stored_embeddings.get_image_paths()

        scores = (torch.matmul(text_embeds, image_embeds.t()) * 100).softmax(dim=1).squeeze().numpy().astype(float)
        best_images = sorted(list(zip(image_paths, scores)), key=lambda x: x[1], reverse=True)[:n]

        return [RankedImage(image_path=path, score=score) for path, score in best_images]
