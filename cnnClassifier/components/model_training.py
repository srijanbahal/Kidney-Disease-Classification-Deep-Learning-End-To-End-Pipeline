# import os
# import urllib.request as request
# from zipfile import ZipFile
# import tensorflow as tf
# import time
# from pathlib import Path
# from cnnClassifier.entity.config_entity import TrainingConfig


# class Training:
#     def __init__(self, config: TrainingConfig):
#         self.config = config

    
#     def get_base_model(self):
#         self.model = tf.keras.models.load_model(
#             self.config.updated_base_model_path
#         )

#     def train_valid_generator(self):

#         datagenerator_kwargs = dict(
#             rescale = 1./255,
#             validation_split=0.20
#         )

#         dataflow_kwargs = dict(
#             target_size=self.config.params_image_size[:-1],
#             batch_size=self.config.params_batch_size,
#             interpolation="bilinear"
#         )

#         valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
#             **datagenerator_kwargs
#         )

#         self.valid_generator = valid_datagenerator.flow_from_directory(
#             directory=self.config.training_data,
#             subset="validation",
#             shuffle=False,
#             **dataflow_kwargs
#         )

#         if self.config.params_is_augmentation:
#             train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
#                 rotation_range=40,
#                 horizontal_flip=True,
#                 width_shift_range=0.2,
#                 height_shift_range=0.2,
#                 shear_range=0.2,
#                 zoom_range=0.2,
#                 **datagenerator_kwargs
#             )
#         else:
#             train_datagenerator = valid_datagenerator

#         self.train_generator = train_datagenerator.flow_from_directory(
#             directory=self.config.training_data,
#             subset="training",
#             shuffle=True,
#             **dataflow_kwargs
#         )

    
#     @staticmethod
#     def save_model(path: Path, model: tf.keras.Model):
#         model.save(path)



    
#     def train(self):
#         self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
#         self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

#         self.model.fit(
#             self.train_generator,
#             epochs=self.config.params_epochs,
#             steps_per_epoch=self.steps_per_epoch,
#             validation_steps=self.validation_steps,
#             validation_data=self.valid_generator
#         )

#         self.save_model(
#             path=self.config.trained_model_path,
#             model=self.model
#         )

# src/components/model_trainer.py

import torch
from torch.utils.data import DataLoader
from torchvision import models, transforms
from pathlib import Path
from cnnClassifier.entity.config_entity import TrainingConfig

class ModelTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_base_model(self):
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.save_model(path=self.config.updated_base_model_path, model=self.model)

    def train_valid_generator(self):
        train_transforms = transforms.Compose([
            transforms.Resize(self.config.params_image_size[:-1]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.train_loader = DataLoader(
            torchvision.datasets.ImageFolder(self.config.training_data, transform=train_transforms),
            batch_size=self.config.params_batch_size,
            shuffle=True
        )

    @staticmethod
    def save_model(path: Path, model: torch.nn.Module):
        torch.save(model.state_dict(), path)

    def train(self):
        self.model.to(self.device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.0001)

        for epoch in range(self.config.params_epochs):
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        self.save_model(path=self.config.trained_model_path, model=self.model)