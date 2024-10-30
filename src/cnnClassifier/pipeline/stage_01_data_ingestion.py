from cnnClassifier.components.data_ingestion import DataIngestion
from cnnClassifier.entity.config_entity import DataIngestionConfig
from cnnClassifier import logger
from cnnClassifier.utils.common import *



STAGE_NAME = "Data Ingestion stage"

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        # create data ingestion config.
        config_file_path = Path(Path.cwd() / 'config/config.yaml')
        params_file_path = Path(Path.cwd() / 'params.yaml')
        data_config_box = read_yaml(config_file_path)

        create_directories([data_config_box.artifacts_root])
        data_ingestion_config = DataIngestionConfig(
            root_dir=data_config_box.data_ingestion.root_dir,
            source_URL=data_config_box.data_ingestion.source_URL,
            local_data_file=data_config_box.data_ingestion.local_data_file,
            unzip_dir=data_config_box.data_ingestion.unzip_dir
        )
        # create data ingestion.
        data_ingestion = DataIngestion(data_ingestion_config)
        # call data ingestion methods.
        base_dir = Path(Path.cwd() / 'artifacts/data_ingestion/dataset')

        data_ingestion.download_file()
        data_ingestion.extract_zip_file()

        train_dir = Path(Path.cwd() / 'artifacts/data_ingestion/dataset/training_data')
        test_dir = Path(Path.cwd() / 'artifacts/data_ingestion/dataset/test_data')
        val_dir = Path(Path.cwd() / 'artifacts/data_ingestion/dataset/validation_data')


        # Create directories for training, testing, and validation
        for split_dir in [train_dir, test_dir, val_dir]:
            os.makedirs(split_dir, exist_ok=True)

        # Define the split sizes
        train_split = 0.7
        val_split = 0.15
        test_split = 0.15

        # Loop through each category folder in the base directory
        for category_folder in os.listdir(base_dir):
            category_path = os.path.join(base_dir, category_folder)
            if os.path.isdir(category_path): #Ensure it's a directory
                data_ingestion.split_data(category_path, train_dir, val_dir, test_dir, train_split, val_split)

        # delete directories which do not contain enough data.
        data_ingestion.delete_folders_not_enough_data()
        # delete empty directories.
        data_ingestion.delete_empty_folders(base_dir)



if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
