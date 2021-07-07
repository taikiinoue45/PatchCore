from omegaconf import OmegaConf

from patchcore.runner import Runner


def test_coverage() -> None:

    cfg = OmegaConf.load("./config.yaml")
    runner = Runner(cfg)
    runner.run()
