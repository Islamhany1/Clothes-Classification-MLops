from dataclasses import dataclass
from pathlib import Path


# @dataclass decorator automatically generates an __init__ method for the class based on the fields you define
# Setting frozen=True ensures that the generated __init__ method won't allow modifications to these attributes after instantiation.
@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    params_image_size: list
    params_learning_rate: float
    params_include_top: bool
    params_weights: str
    params_classes: int