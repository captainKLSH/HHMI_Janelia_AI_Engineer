from src.task2 import logger
from src.task2.config.configuration import ConfigurationManager
from src.task2.components.data_ingestion import DataIngestion
import questionary


STAGE_NAME = "Data Ingestion stage"

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            data_ingestion_config = config.get_data_ingestion_config()
            data_ingestion = DataIngestion(config=data_ingestion_config)
            logger.info("Running data ingestion logic...")
            save_name = None
            while True:
                
                # 2. Create the Interactive Selection Menu
                choice = questionary.select(
                    "What would you like to do?",
                    choices=[
                        "1. Download File as npy",
                        "2. Write Data Dimensions",
                        "3. Exit"
                    ]
                ).ask()
                if choice.startswith("1"):
                    save_name = input("Stage 1. What should I name the output file? (e.g., processed_data.npy): ")
                    if save_name:
                        data_ingestion.download_file(file=save_name)
                        logger.info(f"✅ Downloaded and saved as {save_name}")
                    else:
                        logger.info("⚠️ No filename entered. Operation cancelled.")
                    
            # data_ingestion.download_file_zarr()
                elif choice.startswith("2"):
                    if save_name == None:
                        save_name = input("Input file? (e.g., processed_data.npy): ")
                    
                    logger.info(f"🔍 Checking dimensions for: {save_name}")
                    data_ingestion.data_dim(file=save_name)
                    
                elif choice.startswith("3"):
                    print("Closing pipeline...")
                    break

        except Exception as e:
            print(f"🛑 Error: {e}")
            raise e



if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e