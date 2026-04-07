
from src.task2 import logger
from src.task2.config.configuration import ConfigurationManager
from src.task2.components.multiquery import MultiQueryRetrieval
import questionary


STAGE_NAME = "Visualization Multi Query retrival stage"

class MultiQueryVizPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config=ConfigurationManager()
            mq_config= config.get_mq_config()
            mqr=MultiQueryRetrieval(config=mq_config)
            while True:
                
                choice = questionary.select(
                    "What would you like to do?",
                    choices=[
                        "1. run the grading function for all three strategies(Mean, Score, and RRF)",
                        "2. PCA visualisation to calculates exactly where your specific query mitochondria live",
                        "3. Draw original image with colored boxes, the blank target image, and the individual overlay heatmaps",
                        "4. Exit"
                    ]
                ).ask()
                if choice.startswith("1"):
                    mqr.compare_methods_quantitative()
                elif choice.startswith("2"):
                    mqr.plot_pca_multi_query()
                elif choice.startswith("3"):
                    mqr.plot_multi_query()
                elif choice.startswith("4"):
                    logger.info("Closing Pipeline...")
                    break


        except Exception as e:
            print(f"🛑 Error: {e}")
            raise e



if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = MultiQueryVizPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
    