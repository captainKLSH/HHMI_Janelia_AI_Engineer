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

@dataclass(frozen=True)
class VizConfig:
    root_dir: Path
    dense_embd: Path
    local_data: Path
    outwithin:Path
    out:Path
    z_cord:int
    y_cord: tuple
    x_cord: tuple
    k:int
    th: float
    query_box: bool

@dataclass(frozen=True)
class CrossVizConfig:
    root_dir: Path
    dense_embd: Path
    local_data: Path
    outwithin:Path
    out:Path

@dataclass(frozen=True)
class MultiQueryConfig:
    root_dir: Path
    dense_embd: Path
    local_data: Path
    out:Path
    anot:Path
    k: int