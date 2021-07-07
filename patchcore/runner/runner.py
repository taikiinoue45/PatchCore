from typing import Dict, List, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from albumentations import Compose
from mvtec.builder import Builder
from mvtec.metrics import compute_pro, compute_roc
from mvtec.utils import savegif
from numpy import ndarray
from omegaconf.dictconfig import DictConfig
from scipy.ndimage import gaussian_filter
from sklearn.neighbors import NearestNeighbors
from sklearn.random_projection import SparseRandomProjection
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from patchcore.samplers import KCenterGreedy


class Runner(Builder):
    def __init__(self, cfg: DictConfig) -> None:

        super().__init__()

        self.params: DictConfig = cfg.params
        self.transform: Dict[str, Compose] = {
            key: Compose(self.build_list_cfg(cfg)) for key, cfg in cfg.transform.items()
        }
        self.dataset: Dict[str, Dataset] = {
            key: self.build_dict_cfg(cfg, transform=self.transform[key])
            for key, cfg in cfg.dataset.items()
        }
        self.dataloader: Dict[str, DataLoader] = {
            key: self.build_dict_cfg(cfg, dataset=self.dataset[key])
            for key, cfg in cfg.dataloader.items()
        }
        self.model: Module = self.build_dict_cfg(cfg.model).to(self.params.device)

    def run(self) -> None:

        memory_bank = self._train()
        self._test(memory_bank)

    def _train(self) -> ndarray:

        memory_bank = []
        pbar = tqdm(self.dataloader["train"], desc=f"{self.params.category} - train")
        for _, imgs, _ in pbar:

            with torch.no_grad():
                feature2, feature3 = self.model(imgs.to(self.params.device))
            memory = self._embed_concat(feature2, feature3)
            memory = self._embed_reshape(memory)
            memory_bank.extend(memory.cpu().numpy())

        memory_bank = self._approximate(np.array(memory_bank))
        return memory_bank

    def _test(self, memory_bank: ndarray) -> None:

        artifacts = {"stem": [], "img": [], "mask": [], "ascore": [], "amap": []}
        nbrs = NearestNeighbors(n_neighbors=3, algorithm="ball_tree", metric="minkowski", p=2).fit(
            memory_bank
        )

        self.model.eval()
        pbar = tqdm(self.dataloader["test"], desc=f"{self.params.category} - test")
        for stems, imgs, masks in pbar:

            with torch.no_grad():
                feature2, feature3 = self.model(imgs.to(self.params.device))
            memory = self._embed_concat(feature2, feature3)
            memory = self._embed_reshape(memory)
            memory = memory.cpu().numpy()

            n_dists, _ = nbrs.kneighbors(memory)  # n_dists.shape -> (num_patches, n_neighbors)
            max_n_dist = n_dists[np.argmax(n_dists[:, 0])]  # max_n_dist.shape -> (n_neighbors,)
            w = 1 - (np.max(np.exp(max_n_dist)) / np.sum(np.exp(max_n_dist)))
            ascore = w * max(max_n_dist)

            _, _, h, w = imgs.size()
            amap = n_dists[:, 0].reshape(h // 8, w // 8)
            amap = cv2.resize(amap, (h, w), interpolation=cv2.INTER_LINEAR)
            amap = gaussian_filter(amap, sigma=4)

            artifacts["stem"].extend(stems)
            artifacts["img"].extend(imgs.permute(0, 2, 3, 1).cpu().detach().numpy())
            artifacts["mask"].extend(masks.cpu().detach().numpy())
            artifacts["ascore"].append(ascore)
            artifacts["amap"].append(amap)

        num_data = len(artifacts["ascore"])
        y_preds = np.array(artifacts["ascore"])
        y_trues = np.array(artifacts["mask"]).reshape(num_data, -1).max(axis=1)
        compute_roc(self.params.category, y_trues, y_preds, artifacts["stem"])

        masks = np.array(artifacts["mask"])
        amaps = np.array(artifacts["amap"])
        amaps = (amaps - amaps.min()) / (amaps.max() - amaps.min())
        compute_pro(self.params.category, masks, amaps)

        imgs = self._denormalize(np.array(artifacts["img"]))
        savegif(self.params.category, imgs, masks, amaps)

    def _embed_concat(self, x0: Tensor, x1: Tensor) -> Tensor:

        b0, c0, h0, w0 = x0.size()
        b1, c1, h1, w1 = x1.size()
        x1 = F.interpolate(x1, (h0, w0), mode="bilinear", align_corners=False)
        return torch.cat([x0, x1], dim=1)

    def _embed_reshape(self, embed: Tensor) -> Tensor:

        b, c, h, w = embed.size()
        embed = embed.permute(0, 2, 3, 1).contiguous()
        embed = embed.view(b * h * w, c).contiguous()
        return embed

    def _approximate(self, memory_bank: ndarray) -> ndarray:

        random_projection = SparseRandomProjection(n_components="auto", eps=0.9)
        random_projection.fit(memory_bank)

        sampler = KCenterGreedy(X=memory_bank, y=0, seed=0)
        idxs = sampler.select_batch(model=random_projection, already_selected=[], N=600)

        memory_bank = memory_bank[idxs]
        return memory_bank

    def _denormalize(self, imgs: ndarray) -> ndarray:

        mean = np.array(self.params.normalize_mean)
        std = np.array(self.params.normalize_std)
        imgs = (imgs * std + mean) * 255.0
        return imgs.astype(np.uint8)
