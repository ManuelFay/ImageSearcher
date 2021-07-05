from typing import Tuple, List
import torch

from image_searcher.interfaces.image_loader import ImageLoader
from image_searcher.embedders.clip_embedder import ClipEmbedder


class NaiveSearch:
    """Search with no precomputed embeddings"""
    def __init__(self, image_dir_path: str):
        self.embedder = ClipEmbedder()
        self.loader = ImageLoader(image_dir_path=image_dir_path)
        self.image_paths = self.loader.search_tree()

    def rank_images(self, query: str) -> List[Tuple[str, float]]:
        assert isinstance(query, str)
        text_embeds = self.embedder.embed_text(query)

        image_embeds = None
        for images in self.loader.open_images(self.image_paths):
            new_embeds = self.embedder.embed_images(images)
            image_embeds = torch.cat((image_embeds, new_embeds), dim=0) if image_embeds is not None else new_embeds

        scores = (torch.matmul(text_embeds, image_embeds.t()) * 100).softmax(dim=1).squeeze().numpy()
        return sorted(list(zip(self.image_paths, scores)), key=lambda x: x[1], reverse=True)