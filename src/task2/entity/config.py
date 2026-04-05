from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    scale: str
    out: Path
    z_idx: tuple
    y_idx: tuple
    x_idx: tuple

@dataclass(frozen=True)
class ModelBuildConfig:
    root_dir: Path
    model_name: str
    repo: str
    mw: Path
    hug_mw: str
    HF_save: Path
    localfile: Path
    out: Path
    param_dd: tuple
    patch_size:int
    input_size:int
    n_regis:int
    embed_dim:int
    sli_batch: int
    slide_use:int
    den_sav:Path

