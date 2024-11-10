from cnnClassifier.components.model_trainer import Training
from cnnClassifier import logger
from cnnClassifier.utils.common import *
from cnnClassifier.entity.config_entity import TrainingConfig

STAGE_NAME = "Training"


class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        # create training model config.
        config_file_path = Path(Path.cwd() / 'config/config.yaml')
        params_file_path = Path(Path.cwd() / 'params.yaml')
        data_config_box = read_yaml(config_file_path)
        params_config_box = read_yaml(params_file_path)

        training_config = TrainingConfig(
            root_dir=data_config_box.training.root_dir,
            trained_model_path=data_config_box.training.trained_model_path,
            updated_base_model_path=data_config_box.prepare_base_model.updated_base_model_path,
            training_data=Path(Path.cwd() / "artifacts/data_ingestion/dataset/training_data"),
            params_epochs=params_config_box.EPOCHS,
            params_batch_size=params_config_box.BATCH_SIZE,
            params_is_augmentation=params_config_box.AUGMENTATION,
            params_image_size=params_config_box.IMAGE_SIZE
        )
        # create training model
        training = Training(config=training_config)
        # call the methods.
        training.get_base_model()
        training.train_valid_generator()
        training.train()


if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
