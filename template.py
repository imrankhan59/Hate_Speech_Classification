import os
from pathlib import Path
import logging

logging.basicConfig(level = logging.INFO, format = '[%(asitime)s] : %(message)s :')



list_of_files=[
    ".github/workflows/.gitkeep",
    "src/__init__.py",
    "src/components/__init__.py",
    "src/components/data_ingestion.py",
    "src/components/data_transformation.py",
    "src/components/model_trainer.py",
    "src/components/model_evaluation.py",
    "src/configuration/__init__.py",
    "src/configuration/mongodb_conn.py",
    "src/constant/__init__.py",
    "src/entity/__init__.py",
    "src/entity/config_entity.py",
    "src/entity/artifact_entity.py",
    "src/exception/__init__.py",
    "src/logger/__init__.py",
    "src/pipeline/__init__.py",
    "src/pipeline/training_pipeline.py",
    "src/pipeline/prediction_pipeline.py",
    "src/ml/__init__.py",
    "src/ml/model.py",
    "app.py",
    "demo.py",
    "Dockerfile",
    "setup.py",
    ".dockerignore",
    "src/utils/utils.py",
    "tests/unit/__init__.py",
    "tests/integration/__init__.py",
    "init_setup.sh",
    "requirements.txt",
    "requirements_dev.txt",
    "setup.cfg",
    "pyproject.toml",
    "tox.ini",
    "experiments/eperiments.ipynb",
    ".gitignore",
    ".env",
    "params.yaml",
    "README.md",
    "dvc.yaml",
]


for filepath in list_of_files:
    filepath=Path(filepath)
    filedir,filename=os.path.split(filepath)

    if filedir !="":
        os.makedirs(filedir,exist_ok=True)
        #logging.info(f"Creating directory: {filedir} for file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath)==0):
        with open(filepath,"w") as f:
            pass