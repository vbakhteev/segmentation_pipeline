from models import get_segmentation_model
from .multi_task_pipeline import MultiTaskPipeline


class MultiTaskSegmentation(MultiTaskPipeline):
    def __init__(self, experiment):
        super().__init__(experiment)
        self.model = get_segmentation_model(self.cfg.model)

    @staticmethod
    def pipeline_tasks():
        return ("classification", "segmentation")

