
from src.task2 import logger
from src.task2.config.configuration import ConfigurationManager
from src.task2.components.cross_retrival import CrossDatasetRetrival
import questionary


STAGE_NAME = "Visualization Cross dataset retrival stage"

class CrossVizPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            viz_config = config.get_cross_viz_config()
            cross_ret= CrossDatasetRetrival(config=viz_config)
            while True:
                # 2. Create the Interactive Selection Menu
                choice = questionary.select(
                    "What would you like to do?",
                    choices=[
                        "1. measure domain shift with MMD",
                        "2. PCA visualisation of embedding space",
                        "3. query vector and compute cross-dataset heatmap",
                        "4. Mutual Nearest Neighbours (MNN)",
                        "5. Exit"
                    ]
                ).ask()
                if choice.startswith("1"):
                    cross_ret.mMD()
                elif choice.startswith("2"):
                    cross_ret.plot_embedding_space(n_background=1500)
                elif choice.startswith("3"):
                    cross_ret.plot_cross_retrieval()
                elif choice.startswith("4"):
                    cross_ret.mutual_nn_rate()
                elif choice.startswith("5"):
                    logger.info("Closing Pipeline...")
                    break


        except Exception as e:
            print(f"🛑 Error: {e}")
            raise e



if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = CrossVizPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
    