import os
from pathlib import Path

from cnnClassifier.components.model_evaluation_mlflow import Evaluation
from cnnClassifier import logger
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import read_yaml

STAGE_NAME = "Evaluation stage"


class EvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config_file_path = Path(Path.cwd() / 'config/config.yaml')
        params_file_path = Path(Path.cwd() / 'params.yaml')
        data_config_box = read_yaml(config_file_path)
        params_config_box = read_yaml(params_file_path)

        eval_config = EvaluationConfig(
            path_of_model=Path(os.path.join("artifacts", "training", "model.h5")),
            training_data=Path(os.path.join("artifacts", "data_ingestion", "dataset", "training_data")),
            all_params=params_config_box,
            mlflow_uri="https://dagshub.com/Islamhany1/Clothes-Classification-MLops.mlflow",
            params_image_size=params_config_box.IMAGE_SIZE,
            params_batch_size=params_config_box.BATCH_SIZE
        )
        evaluation = Evaluation(eval_config)
        evaluation.evaluation()
        evaluation.save_score()
        #evaluation.log_into_mlflow()


if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = EvaluationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
