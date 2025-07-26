import os
import urllib.request as request
# from zipfile import ZipFile
import tensorflow as tf
import time
from pathlib import Path
from cnnClassifier.entity.config_entity import TrainingConfig
from cnnClassifier.utils import get_logger
# from tensorflow.keras.models import load_model
from tensorflow.keras import callbacks

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    
    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )
    
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)



    
    def train(self, train_generator, val_generator):
         # Define Callbacks
        lr_callback = callbacks.ReduceLROnPlateau(monitor = 'val_recall', factor = 0.1, patience = 5)
        stop_callback = callbacks.EarlyStopping(monitor = 'val_recall', patience = 5)
        
        EPOCHS = self.config.params_epochs
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=EPOCHS,
            callbacks = [lr_callback, stop_callback]
        )
        get_logger().info(f"Training completed for {EPOCHS} epochs.")
        
        # Save the trained model
        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )

