import os
from pathlib import Path
import random
import shutil
import zipfile
import gdown
from cnnClassifier import logger
from cnnClassifier.utils.common import get_size
from cnnClassifier.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self) -> str:
        '''
        Fetch data from the url
        '''

        try:
            dataset_url = self.config.source_URL
            zip_download_dir = self.config.local_data_file
            os.makedirs("artifacts/data_ingestion", exist_ok=True)
            logger.info(f"Downloading data from {dataset_url} into file {zip_download_dir}")

            file_id = dataset_url.split("/")[-2]
            prefix = 'https://drive.google.com/uc?/export=download&id='
            gdown.download(prefix + file_id, zip_download_dir)

            logger.info(f"Downloaded data from {dataset_url} into file {zip_download_dir}")

        except Exception as e:
            raise e

    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)

    def split_data(self, item_folder, train_dir, val_dir, test_dir, train_split, val_split):
        files = os.listdir(item_folder)
        files = [f for f in files if os.path.isfile(os.path.join(item_folder, f))]
        random.shuffle(files)

        # Determine split sizes
        train_size = int(len(files) * train_split)
        val_size = int(len(files) * val_split)

        # Split files into training, validation, and test sets
        train_files = files[:train_size]
        val_files = files[train_size:train_size + val_size]
        test_files = files[train_size + val_size:]

        # Create category folders in each split directory
        category = os.path.basename(item_folder)
        os.makedirs(os.path.join(train_dir, category), exist_ok=True)
        os.makedirs(os.path.join(val_dir, category), exist_ok=True)
        os.makedirs(os.path.join(test_dir, category), exist_ok=True)

        # Move files
        for file in train_files:
            shutil.move(os.path.join(item_folder, file), os.path.join(train_dir, category, file))

        for file in val_files:
            shutil.move(os.path.join(item_folder, file), os.path.join(val_dir, category, file))

        for file in test_files:
            shutil.move(os.path.join(item_folder, file), os.path.join(test_dir, category, file))

    def delete_folders_not_enough_data(self):
        # Path to the directory you want to delete
        directory_to_delete = ['Blouse', 'Body', 'Skip', 'Top']  # Change this to your directory path
        base_dir = Path(Path.cwd() / 'artifacts/data_ingestion/dataset')
        starting_paths = [Path.joinpath(base_dir, 'training_data'), Path.joinpath(base_dir, 'test_data'), Path.joinpath(base_dir, 'validation_data')]
        # Check if the directory exists
        # Use shutil.rmtree() to delete the directory and its contents
        for starting_path in starting_paths:
            for directory in directory_to_delete:
                if os.path.exists(os.path.join(starting_path, directory)):
                    shutil.rmtree(os.path.join(starting_path, directory))
                    print(f"Directory '{directory}' has been deleted from {starting_path}.")
                else:
                    print(f"Directory '{directory}' does not exist in {starting_path}.")

    def delete_empty_folders(self, path: Path):
        # Recursively go through all directories
        for folder in path.iterdir():
            if folder.is_dir():
                self.delete_empty_folders(folder)  # Recursively check subdirectories
                # Remove the folder if it's empty
                if not any(folder.iterdir()):  # `any(folder.iterdir())` returns False if the folder is empty
                    folder.rmdir()
                    print(f"Deleted empty folder: {folder}")