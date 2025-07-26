import tensorflow as tf
from pathlib import Path
import mlflow
import mlflow.keras
from urllib.parse import urlparse
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import read_yaml, create_directories,save_json


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config


    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)
    

    def evaluation(self, test_generator):
        self.model = self.load_model(self.config.path_of_model)
         # === EVALUATE MODEL ON TEST DATA ===
        test_loss, test_acc, test_precision, test_recall = self.model.evaluate(test_generator)
        self._valid_generator()
        self.score = (test_loss, test_acc, test_precision, test_recall)
        self.save_score()

    def save_score(self):
        scores = {"test-loss": self.score[0], "test-accuracy": self.score[1],"test-precision": self.score[2], "test-recall": self.score[3]}
        save_json(path=Path("scores.json"), data=scores)

    
    def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        
        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
                {"test-loss": self.score[0], "test-accuracy": self.score[1],"test-precision": self.score[2], "test-recall": self.score[3]}
            )
            # Model registry does not work with file store
            if tracking_url_type_store != "file":

                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.keras.log_model(self.model, "model", registered_model_name="ResNet50Model")
            else:
                mlflow.keras.log_model(self.model, "model")
