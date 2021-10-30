# pylint:disable=no-member
from typing import List, Tuple
import logging

import re
import numpy as np
import torch
from tqdm import tqdm

from image_searcher.interfaces.image_loader import ImageLoader
from image_searcher.interfaces.result_interface import RankedImage
from image_searcher.interfaces.stored_embeddings import StoredEmbeddings
from image_searcher.embedders.clip_embedder import ClipEmbedder
from image_searcher.embedders.face_embedder import FaceEmbedder


class Search:
    def __init__(self,
                 image_dir_path: str = None,
                 traverse: bool = False,
                 save_path: str = None,
                 reindex: bool = True,
                 include_faces: bool = False,
                 face_model: str = "large",
                 face_num_jitters: int = 5):

        assert (image_dir_path is not None or save_path is not None), "At least one of the paths " \
                                                                      "(image and save path) needs to be specified"

        self.loader = ImageLoader(image_dir_path=image_dir_path, traverse=traverse)
        self.include_faces = include_faces

        logging.info("Loading CLIP Embedder")
        self.embedder = ClipEmbedder()
        self.face_embedder = FaceEmbedder(model=face_model, num_jitters=face_num_jitters)

        logging.info("Loading pre-computed embeddings")
        self.stored_embeddings = StoredEmbeddings(save_path=save_path if save_path else image_dir_path)
        logging.info(f"{len(self.stored_embeddings.get_image_paths())} files are indexed.")

        if reindex:
            logging.info(f"Re-indexing the image files in {image_dir_path}")
            self.reindex()

        self.stored_embeddings.set_embedding_tensor()

        if image_dir_path:
            logging.info(f"Excluding files from search if not in {image_dir_path}")
            self.stored_embeddings.exclude_extra_files(filter_path=image_dir_path)

        if include_faces:
            logging.info("Recomputing face embedding matrix")
            self.stored_embeddings.set_all_face_embeddings()

        logging.info("Setup over, Searcher is ready to be queried")

    def reindex(self):
        waiting_list = set(self.loader.search_tree()) - set(self.stored_embeddings.get_image_paths())
        if not waiting_list:
            return

        for idx, image_path in enumerate(tqdm(waiting_list)):
            self.index_image(image_path)
            if idx % 200 == 0:
                self.stored_embeddings.update_file()

        self.stored_embeddings.update_file()

    def index_image(self, image_path):
        try:
            images = [self.loader.open_image(image_path)]
            self.stored_embeddings.add_embedding(image_path, self.embedder.embed_images(images))
            if self.include_faces:
                self.stored_embeddings.add_face_info(image_path, self.face_embedder.embed_image(images[0]))

        except Exception as exception:
            logging.warning(f"Image {image_path} has failed to process - adding it to fail list.")
            self.stored_embeddings.add_embedding(image_path, torch.zeros((1, 512)))
            logging.warning(exception)

    @staticmethod
    def parse_query(query) -> Tuple[str, List[str]]:
        """
        Parse query and find tags
        :param query: input query
        :return: Seaparated query and list of tags
        """
        matches = re.findall(r"\B(\#[a-zA-Z]+\b)", query)
        for match in matches:
            query = query.replace(match, "")
        return query.strip(), [tag[1:] for tag in matches]

    def filter_images(self, tags):
        """
        Filter the search pool based on predetermined tags:
        Supported tags are:
        - #groups: WIP: Group pictures (more than 5 persons)
        - #{category}: Amongst "screenshot", "drawing", "photo", "schema", "selfie"

        :param tags: Filtering tags from a list
        :return: image_embeds, image_paths to feed into the ranking system
        """
        image_embeds, image_paths = self.stored_embeddings.get_embedding_tensor()
        mask = torch.ones((len(image_paths)), dtype=torch.bool)

        categories = ["screenshot", "drawing", "photo", "schema", "selfie"]
        for tag in tags:
            if tag in categories:
                tag_embed = self.embedder.embed_text(f"This image is a {tag}")
                categories.remove(tag)
                opposite_tags_embed = [self.embedder.embed_text(f"This image is a {op_tag}") for op_tag in categories]
                for opposite_tag_embed in opposite_tags_embed:
                    mask = mask & (torch.matmul(tag_embed, image_embeds.t()) > torch.matmul(opposite_tag_embed,
                                                                                            image_embeds.t())).squeeze()
                logging.info(f"Filtered non {tag} images")

            if tag == "group" and self.face_embedder:
                _, idx2path = self.stored_embeddings.get_all_face_embeddings()
                mask = mask & torch.Tensor([idx2path.count(image_path) > 4 for image_path in image_paths]).bool()

        if mask.sum().item() == 0:
            logging.warning("Tags filtered out all original pictures. Filtering desactivated.")
            return image_embeds, image_paths

        return image_embeds[mask], list(np.array(image_paths)[mask.numpy()])

    def rank_images(self, query: str, n: int = 10) -> List[RankedImage]:
        assert isinstance(query, str)
        query, tags = self.parse_query(query)
        text_embeds = self.embedder.embed_text(query)
        image_embeds, image_paths = self.filter_images(tags)
        if len(image_paths) == 1:
            return [RankedImage(image_path=image_paths[0], score=1)]
        scores = (torch.matmul(text_embeds, image_embeds.t()) * 100).softmax(dim=1).squeeze().numpy().astype(float)
        best_images = sorted(list(zip(image_paths, scores)), key=lambda x: x[1], reverse=True)[:n]

        return [RankedImage(image_path=path, score=score) for path, score in best_images]

    def rank_images_by_faces(self, image_path: str, n: int = 10) -> List[List[RankedImage]]:
        assert isinstance(image_path, str)
        best_faces = []

        try:
            all_face_embeddings, idx2path = self.stored_embeddings.get_all_face_embeddings()
            face_embeds = self.stored_embeddings.embeddings[image_path]["face_embeddings"]

            for embed in face_embeds:
                scores = torch.linalg.norm(torch.Tensor([embed]) - all_face_embeddings, dim=1).numpy().astype(float)
                best_images: List = sorted(list(zip(idx2path, scores)), key=lambda x: x[1], reverse=False)[:n]
                best_faces.append([RankedImage(image_path=path, score=score) for path, score in best_images])

            return best_faces

        except KeyError:
            return []
