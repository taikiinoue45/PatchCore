import numpy as np
import torch
import torch.nn.functional as F
from mvtec.builder import Builder
from numpy import ndarray
from omegaconf.dictconfig import DictConfig
from sklearn.neighbors import NearestNeighbors
from sklearn.random_projection import SparseRandomProjection
from torch import Tensor
from tqdm import tqdm

from patchcore.samplers import KCenterGreedy


class Runner(Builder):
    def __init__(self, cfg: DictConfig) -> None:

        super().__init__()

    def run(self) -> None:

        memory_bank = self._train()
        self._test(memory_bank)

    def _train(self) -> ndarray:

        memory_bank = []
        pbar = tqdm(self.dataloader["train"], desc=f"{self.params.category} - train")
        for _, imgs, _ in pbar:

            feature2, feature3 = self.model(imgs.to(self.params.device))
            memory = self._embed_concat(feature2, feature3)
            memory = self._embed_reshape(memory)
            memory_bank.extend(memory)

        memory_bank = self._random_projection(memory_bank)
        memory_bank = self._coreset_sampling(memory_bank)
        return memory_bank

    def _test(self, memory_bank: ndarray) -> None:

        artifacts = {"stem": [], "img": [], "mask": [], "ascore": []}
        nbrs = NearestNeighbors(n_neighbors=3, algorithm="ball_tree", metric="minkowski", p=2).fit(
            memory_bank
        )

        self.model.eval()
        pbar = tqdm(self.dataloader["test"], desc=f"{self.params.category} - test")
        for stems, imgs, masks in pbar:

            feature2, feature3 = self.model(imgs.to(self.params.device))
            memory = self._embed_concat(feature2, feature3)
            memory = self._embed_reshape(memory)
            memory = self._random_projection(memory)

            n_dists, _ = nbrs.kneighbors(memory)  # n_dists.shape -> (num_patches, n_neighbors)
            max_n_dist = n_dists[np.argmax(n_dists[:, 0])]  # max_n_dist.shape -> (n_neighbors,)
            w = 1 - (np.max(np.exp(max_n_dist)) / np.sum(np.exp(max_n_dist)))
            ascore = w * max(max_n_dist)

            artifacts["stem"].extend(stems)
            artifacts["img"].extend(imgs.permute(0, 2, 3, 1).cpu().detach().numpy())
            artifacts["mask"].extend(masks.cpu().detach().numpy())
            artifacts["ascore"].extend(ascore)

    def _embed_concat(self, x0: Tensor, x1: Tensor) -> Tensor:

        b0, c0, h0, w0 = x0.size()
        b1, c1, h1, w1 = x1.size()
        x1 = F.interpolate(x1, (h0, w0), mode="bilinear")
        return torch.cat([x0, x1], dim=1)

    def _embed_reshape(self, embed: Tensor) -> Tensor:

        b, c, h, w = embed.size()
        embed = embed.permute(1, 0, 2, 3)
        embed = embed.view(c, b * h * w)
        return embed

    def _random_projection(self, embed: ndarray) -> ndarray:

        random_projection = SparseRandomProjection(n_components="auto", eps=0.9)
        embed = random_projection.fit_transform(embed)
        return embed

    def _coreset_sampling(self, embed: ndarray) -> ndarray:

        sampler = KCenterGreedy(X=embed, y=0, seed=0)
        idxs = sampler.select_batch(model=None, already_selected=[], N=600)
        embed = embed[idxs]
        return embed
