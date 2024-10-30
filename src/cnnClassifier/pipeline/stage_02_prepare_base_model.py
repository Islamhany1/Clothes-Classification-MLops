from cnnClassifier.components.prepare_base_model import PrepareBaseModel
from cnnClassifier.utils.common import *
from cnnClassifier import logger
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig

STAGE_NAME = "Prepare base model"


class PrepareBaseModelTrainingPipeline:
    def __init__(self):
        pass

    @staticmethod
    def main():
        # create prepare base model config.
        config_file_path = Path(Path.cwd() / 'config/config.yaml')
        params_file_path = Path(Path.cwd() / 'params.yaml')
        data_config_box = read_yaml(config_file_path)
        params_config_box = read_yaml(params_file_path)

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(data_config_box.prepare_base_model.root_dir),
            base_model_path=Path(data_config_box.prepare_base_model.base_model_path),
            updated_base_model_path=Path(data_config_box.prepare_base_model.updated_base_model_path),
            params_image_size=params_config_box.IMAGE_SIZE,
            params_learning_rate=params_config_box.LEARNING_RATE,
            params_include_top=params_config_box.INCLUDE_TOP,
            params_weights=params_config_box.WEIGHTS,
            params_classes=params_config_box.CLASSES
        )
        # create base model using the base model config.
        prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
        # call the methods.
        prepare_base_model.get_base_model()
        prepare_base_model.update_base_model()


if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = PrepareBaseModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
