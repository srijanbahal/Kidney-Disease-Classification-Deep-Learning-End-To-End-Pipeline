import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from pathlib import Path
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from cnnClassifier.utils import get_logger
from tensorflow.keras.models import Model
from tensorflow.keras import callbacks
from tensorflow.keras.optimizers import Adam


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    
    def load_base_model(self, model_name):
        self.config.base_model_path = Path(self.config.base_model_path)
        self.config.model_name = model_name
        IMG_SIZE = self.config.params_image_size
        params = self.config.params
        
        if model_name == "resnet50":
            model = ResNet50(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
            # Modify the final layer for num_classes
            self.save_model(path=self.config.base_model_path, model=model)
            return model
        elif model_name == "swin-transformer":
            pass

        else:
            raise ValueError(f"Unknown model: {model_name}")

        # self.save_model(path=self.config.base_model_path, model=self.model)
    

    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        if freeze_all:
            for layer in model.layers:
                model.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                model.trainable = False

                
        # === ADD CUSTOM LAYERS ===
        x = model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation="relu")(x)
        x = Dense(128, activation="relu")(x)
        x = Dense(64, activation="relu")(x)
        x = Dense(len(classes.unique()), activation="softmax")(x)  # Output layer with # of classes

                
        # === CREATE FINAL MODEL ===
        full_model = Model(inputs=model.input, outputs=x)

            # Define Callbacks
        lr_callback = callbacks.ReduceLROnPlateau(monitor = 'val_recall', factor = 0.1, patience = 5)
        stop_callback = callbacks.EarlyStopping(monitor = 'val_recall', patience = 5)   
                
        
        # === COMPILE MODEL ===
        full_model.compile(
                optimizer=Adam(learning_rate=0.0001),
                loss="categorical_crossentropy",
                metrics=[
                    "accuracy",
                    tf.keras.metrics.Precision(name="precision"),
                    tf.keras.metrics.Recall(name="recall")
                ]
            )

        full_model.summary()
        get_logger().info(f"Model {model.name} prepared with {len(classes.unique())} classes.")
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
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

