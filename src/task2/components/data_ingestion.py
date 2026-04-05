import os
import numpy as np
import fibsem_tools as fst
import xarray as xr
import s3fs
from pathlib import Path
from src.task2 import logger
from src.task2.utils.common import get_size
import datetime
import zarr
from src.task2.entity.config import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    
    def download_file(self, file= 'chunk.npy')-> str:
        '''
        Fetch data from the url
        '''

        try: 
            dataset_url = self.config.source_URL
            local_file_path= os.path.join(self.config.local_data_file,file)
            scale=self.config.scale

            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            # 2. Check if file already exists to save time/bandwidth
            if not os.path.exists(local_file_path):
                logger.info(f"Connecting to s3 {dataset_url}...")

                tree = fst.read_xarray(dataset_url, storage_options={'anon': True})
                
                data_array = tree[scale].data
                logger.info(f"Fetching chunks from S3 {scale}...")
                # Based on your metadata chunksize of 448
                z_range = slice(self.config.z_idx[0], self.config.z_idx[1])       # 2 vertical chunks
                y_range = slice(self.config.y_idx[0], self.config.y_idx[1])   # 1 chunk wide
                x_range = slice(self.config.x_idx[0], self.config.x_idx[1])   # 1 chunk deep

                # .compute() pulls the data from S3 into memory
                volume = data_array[z_range, y_range, x_range].compute()
                # 5. Save as .npy for fast loading into DINO v3
                np.save(local_file_path, volume)
                logger.info(f"Download complete! Volume {volume.shape} saved to: {local_file_path} (Size: {get_size(Path(local_file_path))})")
                with open(self.config.out, 'a') as f:
                    f.write("-"*30+'\n')
                    f.write("-"*30+'\n')
                    f.write("-"*30+'\n')
                    f.write(f"Node {scale} Value: {tree[scale]}\n")
                    f.write("-"*30+'\n')
            else:
                logger.info(f"File already exists at: {local_file_path} (Size: {get_size(Path(local_file_path))})")
        except Exception as e:
            logger.error(f"Error occurred while downloading: {e}")
            raise e
    # def download_file_zarr(self)-> str:
    #     try:
    #         fs = s3fs.S3FileSystem(anon=True)
    #         scale = self.config.scale
    #         s3_path = os.path.join(self.config.source_URL,scale)
    #         local_path = Path(self.config.local_data_file)
    #         if not local_path.exists():
    #             logger.info(f"Connecting to Zarr V2 at: {s3_path}")
    #             store = s3fs.S3Map(root=s3_path, s3=fs, check=False)
    #             z_array = zarr.open(store, mode='r')
    #             ds = xr.DataArray(
    #                 z_array, 
    #                 dims=("z", "y", "x"), 
    #                 name="t_cell_em"
    #             )
    #             roi = ds.isel(
    #             z=slice(4872, 5320), 
    #             y=slice(587, 1035), 
    #             x=slice(3487, 3935)
    #             )
    #             logger.info(f"Computing Dask graph for {scale} ROI...")
    #             volume_np = roi.values
    #             local_path.parent.mkdir(parents=True, exist_ok=True)
    #             np.save(local_path, volume_np)
    #             logger.info(f"ROI saved successfully to {local_path}")
    #         return str(local_path)
    #     except Exception as e:
    #         logger.error(f"Lazy download failed: {e}")
    #         raise e


    def data_dim(self,file='chunk.npy'):
        local_file_path= os.path.join(self.config.local_data_file,file)
        data = np.load(local_file_path, mmap_mode='r')
        dd={
            "Shape": list(data.shape), 
            "Dtype": str(data.dtype), 
            "Total Elements": int(data.size),
            "Memory Size MB": float(f"{data.nbytes / (1024**2):.2f}")
        }
        logger.info("writing Data Dimensions")
        with open(self.config.out, 'a') as f:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"\n--- Analysis Run: {timestamp} ---\n")
            f.write(f"Data Summary for {local_file_path}\n")
            f.write("-" * 30 + "\n")
            # for key, value in dd.items():
            #     f.write(f"{key}: {value}\n")
            f.write(f"Shape: {dd['Shape']}\n")
            f.write(f"Dtype: {dd['Dtype']}\n")
            f.write(f"Total Elements: {dd['Total Elements']:,}\n")
            f.write(f"Memory: {dd['Memory Size MB']} MB\n")
            f.write("-" * 30 + "\n")
        logger.info("Created and stored Data Summary")