# Here I will  take Data from Artifacts/dataset and split it into train, validation, and test sets.

# And the Data Will be trnaformed and preprocessed and saved in Artifacts/preprocessed_data
import os
import tensorflow as tf
from cnnClassifier.entity.config_entity import TrainValTestConfig
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.resnet50 import preprocess_input

class TrainValTest:
    def __init__(self, config):
        self.config = config

    def load_data(self):
        """
        This method will load the data from the specified path.
        """
        # Implement the logic to load the data
        dataset_path = "artifacts/raw"  # Change this to your dataset directory
        output_dir = "/artifacts/splits"  # Directory to store train, val, test split
        os.makedirs(output_dir, exist_ok=True)

        # === CREATING TRAIN, VALIDATION, TEST SPLIT ===
        all_images = []
        all_labels = []

        for class_name in os.listdir(dataset_path):  
            class_path = os.path.join(dataset_path, class_name)
            if os.path.isdir(class_path):
                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_name)
                    all_images.append(img_path)
                    all_labels.append(class_name)
        df = pd.DataFrame({"image": all_images, "label": all_labels})
        # Split dataset: 70% Train, 15% Val, 15% Test
        train_df, temp_df = train_test_split(df, test_size=0.4, stratify=df["label"], random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42)
        
        # Save splits as CSV
        train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
        val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
        test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
        print(f"Saved train, val, test splits to {output_dir}")
        return 


    def preprocess_data(self, train_df, val_df, test_df):
        """
        This method will preprocess the data.
        """
        IMG_SIZE = self.config.params_image_size
        BATCH_SIZE = self.config.params_batch_size
    
        # === IMAGE DATA GENERATORS WITH AUGMENTATION ===
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,  # ResNet50 specific preprocessing
            rotation_range=20,        # Random rotation (0-20 degrees)
            width_shift_range=0.2,    # Random width shift (20% of width)
            height_shift_range=0.2,   # Random height shift (20% of height)
            zoom_range=0.2,           # Random zoom
            horizontal_flip=True      # Random horizontal flip
        )
                
        val_test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)  # No augmentation for val/test

        train_generator = train_datagen.flow_from_dataframe(
            train_df,
            x_col="image",
            y_col="label",
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            class_mode="categorical"
        )

        val_generator = val_test_datagen.flow_from_dataframe(
            val_df,
            x_col="image",
            y_col="label",
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            class_mode="categorical"
        )

        test_generator = val_test_datagen.flow_from_dataframe(
            test_df,
            x_col="image",
            y_col="label",
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            class_mode="categorical",
            shuffle=False  # Important for evaluation
        )
        pass
        return train_generator, val_generator, test_generator