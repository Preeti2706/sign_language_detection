import os, sys, zipfile, shutil, yaml
from signLanguage.utils.main_utils import read_yaml_file
from signLanguage.logger import logging
from signLanguage.exception import SignException
from signLanguage.entity.config_entity import ModelTrainerConfig
from signLanguage.entity.artifacts_entity import ModelTrainerArtifact


class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig):
        self.model_trainer_config = model_trainer_config

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")

        try:
            logging.info("Unzipping data...")

            # ✅ unzip using Python instead of os.system
            if os.path.exists("Sign_language_data.zip"):
                with zipfile.ZipFile("Sign_language_data.zip", 'r') as zip_ref:
                    zip_ref.extractall(".")
                os.remove("Sign_language_data.zip")
            else:
                raise FileNotFoundError("Sign_language_data.zip not found!")

            # ✅ check if data.yaml exists
            if not os.path.exists("data.yaml"):
                raise FileNotFoundError("data.yaml not found after unzipping!")

            with open("data.yaml", 'r') as stream:
                num_classes = str(yaml.safe_load(stream)['nc'])

            model_config_file_name = self.model_trainer_config.weight_name.split(".")[0]
            logging.info(f"Base model config: {model_config_file_name}")

            config = read_yaml_file(f"yolov5/models/{model_config_file_name}.yaml")
            config['nc'] = int(num_classes)

            # save custom model config
            custom_cfg_path = f'yolov5/models/custom_{model_config_file_name}.yaml'
            with open(custom_cfg_path, 'w') as f:
                yaml.dump(config, f)

            # ✅ train model
            os.system(
                f"cd yolov5 && python train.py "
                f"--img 416 --batch {self.model_trainer_config.batch_size} "
                f"--epochs {self.model_trainer_config.no_epochs} "
                f"--data ../data.yaml "
                f"--cfg ./models/custom_{model_config_file_name}.yaml "
                f"--weights {self.model_trainer_config.weight_name} "
                f"--name yolov5s_results --cache"
            )

            # copy trained model
            best_model_src = "yolov5/runs/train/yolov5s_results/weights/best.pt"
            best_model_dst = os.path.join(self.model_trainer_config.model_trainer_dir, "best.pt")
            os.makedirs(self.model_trainer_config.model_trainer_dir, exist_ok=True)
            shutil.copy(best_model_src, best_model_dst)

            # cleanup
            shutil.rmtree("yolov5/runs", ignore_errors=True)
            for folder in ["train", "test"]:
                shutil.rmtree(folder, ignore_errors=True)
            if os.path.exists("data.yaml"):
                os.remove("data.yaml")

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=best_model_dst,
            )

            logging.info("Exited initiate_model_trainer method of ModelTrainer class")
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")

            return model_trainer_artifact

        except Exception as e:
            raise SignException(e, sys)