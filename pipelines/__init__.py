from .segmentator import Segmentator

available_pipelines = {"segmentation": Segmentator}


def get_pipeline(cfg):
    name = cfg.model.pipeline
    if name not in available_pipelines:
        raise KeyError(f"Pipeline {name} is not supported")

    return available_pipelines[name]
