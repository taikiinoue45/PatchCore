from mvtec.builder import Builder
from omegaconf.dictconfig import DictConfig


class Runner(Builder):
    def __init__(self, cfg: DictConfig) -> None:

        super().__init__()
