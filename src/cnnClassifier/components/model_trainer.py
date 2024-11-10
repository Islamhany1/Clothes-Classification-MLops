import tensorflow as tf
from keras.src.legacy.preprocessing.image import ImageDataGenerator

from cnnClassifier.entity.config_entity import TrainingConfig
from pathlib import Path


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )




    def train_valid_generator(self):
        # Define paths to your directories
        train_dir = Path(Path.cwd() / 'artifacts/data_ingestion/dataset/training_data')
        test_dir = Path(Path.cwd() / 'artifacts/data_ingestion/dataset/test_data')
        val_dir = Path(Path.cwd() / 'artifacts/data_ingestion/dataset/validation_data')

        # Define image parameters
        image_size = (224, 224)  # ResNet typically uses 224x224 image size
        batch_size = 32

        # Create ImageDataGenerator objects for augmentation and rescaling

        # Augmentation for training data
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,  # Normalize pixel values to [0, 1]
            rotation_range=30,  # Rotate images randomly within 30 degrees
            width_shift_range=0.2,  # Shift images horizontally by 20% of the width
            height_shift_range=0.2,  # Shift images vertically by 20% of the height
            shear_range=0.2,  # Shear angle for augmentation
            zoom_range=0.2,  # Random zoom into images by 20%
            horizontal_flip=True,  # Flip images horizontally
            fill_mode='nearest'  # Fill empty pixels after a transformation
        )

        # Rescale validation and test data (no augmentation)
        val_test_datagen = ImageDataGenerator(rescale=1. / 255)

        # Load and augment training data
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=image_size,  # ResNet expects 224x224 input size
            batch_size=batch_size,
            class_mode='categorical'  # Use 'categorical' for multi-class classification
        )

        # Load validation data (no augmentation)
        validation_generator = val_test_datagen.flow_from_directory(
            val_dir,
            target_size=image_size,
            batch_size=batch_size,
            class_mode='categorical'
        )

        # Load test data (no augmentation)
        test_generator = val_test_datagen.flow_from_directory(
            test_dir,
            target_size=image_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False  # Don't shuffle test data for accurate results
        )

        self.train_generator = train_generator
        self.valid_generator = validation_generator

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

    def train(self):
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator
        )

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )
