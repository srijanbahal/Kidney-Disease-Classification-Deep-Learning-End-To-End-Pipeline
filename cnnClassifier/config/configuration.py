from cnnClassifier.constants import *
import os
from cnnClassifier.utils.common import read_yaml, create_directories,save_json
from cnnClassifier.entity.config_entity import (DataIngestionConfig,
                                                PrepareBaseModelConfig,
                                                TrainingConfig,
                                                EvaluationConfig)



class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            # unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    

    def get_prepare_base_model_config(self, model_name: str) -> PrepareBaseModelConfig:
        """
        Takes a model name (e.g., 'resnet50') and returns the configuration
        for preparing that specific base model.
        """
        try:
            model_config = self.params.MODELS[model_name]
            common_params = model_config.PARAMS
        except KeyError:
            raise ValueError(f"Configuration for model '{model_name}' not found in params.yaml")

        config = self.config.prepare_base_model
        
        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            
            # Use the dynamically selected parameters
            model_name=model_config.NAME,
            params_image_size=common_params.IMAGE_SIZE,
            params_include_top=common_params.INCLUDE_TOP,
            params_weights=common_params.WEIGHTS,
            
            # Global param
            params_classes=self.params.CLASSES
        )

        return prepare_base_model_config


    def get_training_config(self, model_name: str, experiment_name: str) -> TrainingConfig:
        """
        Gets the training config by merging common model params with specific
        experiment params.
        """
        try:
            model_config = self.params.MODELS[model_name]
            experiment_params = model_config.EXPERIMENTS[experiment_name]
            common_params = model_config.PARAMS
        except KeyError:
            raise ValueError(f"Experiment '{experiment_name}' for model '{model_name}' not found.")

        training = self.config.training
        prepare_base_model = self.config.prepare_base_model
        training_data = os.path.join(self.config, "kidney-ct-scan-image")

        create_directories([Path(training.root_dir)])

        # Create the config by combining params from different sections
        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            updated_base_model_path=Path(prepare_base_model.updated_base_model_path),
            training_data=Path(training_data),
            
            # Get params from the specific experiment
            params_epochs=experiment_params.EPOCHS,
            params_batch_size=experiment_params.BATCH_SIZE,
            params_use_class_weights=experiment_params.USE_CLASS_WEIGHTS,
            
            # Get params from the common model section
            params_image_size=common_params.IMAGE_SIZE,
            
            # Get global param
            params_is_augmentation=self.params.AUGMENTATION 
        )

        return training_config

    def get_evaluation_config(self) -> EvaluationConfig:
        eval_config = EvaluationConfig(
            path_of_model="artifacts/training/model.h5",
            training_data="artifacts/data_ingestion/kidney-ct-scan-image",
            mlflow_uri="https://dagshub.com/entbappy/Kidney-Disease-Classification-MLflow-DVC.mlflow",
            all_params=self.params,
            params_image_size=self.params.IMAGE_SIZE,
            params_batch_size=self.params.BATCH_SIZE
        )
        return eval_config

