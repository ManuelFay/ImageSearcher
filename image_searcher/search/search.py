from typing import Tuple, List
import logging

import torch
from tqdm import tqdm

from image_searcher.interfaces.image_loader import ImageLoader
from image_searcher.interfaces.stored_embeddings import StoredEmbeddings
from image_searcher.embedders.clip_embedder import ClipEmbedder


class Search:
    """Search with no precomputed embeddings"""
    def __init__(self, image_dir_path: str, traverse: bool = False):
        self.loader = ImageLoader(image_dir_path=image_dir_path, traverse=traverse)

        logging.info("Loading CLIP Embedder")
        self.embedder = ClipEmbedder()

        logging.info("Loading pre-computed embeddings")
        self.stored_embeddings = StoredEmbeddings(save_path=image_dir_path)

        logging.info(f"Re-indexing the image files in {image_dir_path}")
        self.reindex()

    def reindex(self):
        waiting_list = set(self.loader.search_tree()) - set(self.stored_embeddings.get_image_paths())
        if not waiting_list:
            return

        # TODO: Optimize (batch, use dataloader, remove custom batching)
        for image_path in tqdm(waiting_list):
            try:
                image = list(*self.loader.open_images([image_path]))
                self.stored_embeddings.add_embedding(image_path, self.embedder.embed_images(image))
            except Exception as exception:
                logging.warning(f"Image {image_path} has failed to process - adding it to blacklist.")
                self.stored_embeddings.add_embedding(image_path, torch.zeros((1, 512)))
                logging.warning(exception)

        self.stored_embeddings.update_file()

    def rank_images(self, query: str) -> List[Tuple[str, float]]:
        assert isinstance(query, str)
        text_embeds = self.embedder.embed_text(query)
        image_embeds = self.stored_embeddings.get_embedding_tensor()

        scores = (torch.matmul(text_embeds, image_embeds.t()) * 100).softmax(dim=1).squeeze().numpy()
        return sorted(list(zip(self.stored_embeddings.get_image_paths(), scores)), key=lambda x: x[1], reverse=True)
