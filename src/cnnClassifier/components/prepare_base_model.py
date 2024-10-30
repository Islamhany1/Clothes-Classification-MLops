from pathlib import Path
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig
import keras.layers
import keras.applications
import keras.optimizers
import keras.losses


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config
        self.model = None
        self.full_model = None

    # loads the pre-trained VGG16 model.
    def get_base_model(self):
        self.model = keras.applications.vgg16.VGG16(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top
        )

        self.save_model(path=self.config.base_model_path, model=self.model)

    @staticmethod
    # Modifies the base VGG16 by adding a custom classification head.
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        if freeze_all:  # to freeze all layers
            for _ in model.layers:
                model.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):  # freeze only some of them
            for _ in model.layers[:-freeze_till]:
                model.trainable = False

        flatten_in = keras.layers.Flatten()(model.output)
        prediction = keras.layers.Dense(
            units=classes,
            activation="softmax"
        )(flatten_in)

        full_model = keras.models.Model(
            inputs=model.input,
            outputs=prediction
        )

        full_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )

        full_model.summary()
        return full_model

    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )

        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)

    @staticmethod
    def save_model(path: Path, model: keras.Model):
        model.save(path)
