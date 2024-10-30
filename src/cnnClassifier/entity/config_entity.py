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