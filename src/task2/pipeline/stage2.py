from src.task2 import logger
from src.task2.config.configuration import ConfigurationManager
from src.task2.components.modelbuild import ModelBuild
import questionary


STAGE_NAME = "Model Building stage"

class ModelBuildingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            model_enc_config = config.get_model_build_config()
            diagnosis = ModelBuild(config=model_enc_config)
            device=diagnosis.sysConfig()
            
            device = "cpu"
            model = None
            bb = None
            file = None
            while True:
                
                # 2. Create the Interactive Selection Menu
                choice = questionary.select(
                    "What would you like to do?",
                    choices=[
                        "1. Load Hugging Face Model",
                        "2. Load Local Model",
                        "3. Generate Dense Embeddings",
                        "4. Visualize PCA",
                        "5. Exit"
                    ]
                ).ask()
                if choice.startswith("1"):
                    model, bb=diagnosis.loadFromHuggingFace(device=device)
                elif choice.startswith("2"):
                    model, bb=diagnosis.loadFromLocal(device=device)  
                elif choice.startswith("3"):
                    if model is None:
                        logger.info("❌ Error: Please load a model first (Option 2 or 3).")
                    else:
                        file=input(f"Input data file to get embeddings for (e.g. liver.npy):")
                        diagnosis.get_dense_embeddings(device=device,model=model,backbone=bb, file=file)
                        logger.info("✅ Embeddings calculated.")
                elif choice.startswith("4"):
                    slice_idx = questionary.text("Slice index?", default="12").ask()
                    diagnosis.visualize_embeddings_pca(slice_idx= int(slice_idx))
                elif choice.startswith("5"):
                    logger.info("Closing Pipeline...")
                    break

        except Exception as e:
            print(f"🛑 Error: {e}")
            raise e



if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelBuildingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e