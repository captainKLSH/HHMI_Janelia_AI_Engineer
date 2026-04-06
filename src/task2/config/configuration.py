from src.task2 import logger
from src.task2.constants import *
from src.task2.utils.common import read_yaml, create_directories

from src.task2.entity.config import (DataIngestionConfig,
                                     ModelBuildConfig,
                                     VizConfig,
                                     CrossVizConfig
                                     )

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
            root_dir= Path(config.root_dir),
            source_URL= str(config.source_URL),
            local_data_file=Path(config.local_data_file),
            scale= str(self.params.scale),
            out= Path(config.OUTPUT),
            z_idx= tuple(self.params.z_idx),
            y_idx= tuple(self.params.y_idx),
            x_idx= tuple(self.params.x_idx),

        )
        return data_ingestion_config
    
    def get_model_build_config(self) -> ModelBuildConfig:
        config = self.config.model_build

        create_directories([config.root_dir])

        model_build_config = ModelBuildConfig(
            root_dir=Path(config.root_dir),
            model_name= str(config.model_name),
            repo=str(config.REPO),
            mw=Path(config.M_weight),
            hug_mw=str(config.Hug_MW),
            HF_save=Path(config.Hug_save),
            out= Path(config.OUTPUT),
            param_dd= tuple(self.params.input_size_test),
            patch_size= int(self.params.patch_size),
            input_size=int(self.params.INPUT_SIZE),
            n_regis=int(self.params.N_REGISTERS),
            embed_dim=int(self.params.EMBED_DIM),
            localfile=Path(config.local_data_file),
            sli_batch=int(self.params.Slice_BATCH_size),
            slide_use=int(self.params.Slice_use),
            den_sav=Path(config.Dense_embedding)
        )

        return model_build_config
    def get_viz_config(self) -> VizConfig:
        config = self.config.viz

        create_directories([config.root_dir])

        viz_config = VizConfig(
            root_dir=Path(config.root_dir),
            dense_embd=Path(config.Dense_embedding),
            local_data=Path(config.local_data_file),
            outwithin= Path(config.OUTPUT_within),
            out=Path(config.OUTPUT),
            z_cord=int(self.params.z_cord),
            y_cord=tuple(self.params.y_cord),
            x_cord=tuple(self.params.x_cord),
            k=int(self.params.k),
            th=float(self.params.threshold),
            query_box=bool(self.params.query_box),
        )
        return viz_config
    
    def get_cross_viz_config(self) -> CrossVizConfig:
        config = self.config.viz
        create_directories([config.root_dir])
        cross_viz_config = CrossVizConfig(
            root_dir=Path(config.root_dir),
            dense_embd=Path(config.Dense_embedding),
            local_data=Path(config.local_data_file),
            outwithin= Path(config.OUTPUT_within),
            out=Path(config.OUTPUT)
        )
        return cross_viz_config