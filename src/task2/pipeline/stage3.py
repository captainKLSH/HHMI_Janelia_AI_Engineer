from src.task2 import logger
from src.task2.config.configuration import ConfigurationManager
from src.task2.components.within_retrival import WithinDatasetRetrival
import questionary


STAGE_NAME = "Visualization retrival stage"
class VizPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            viz_config = config.get_viz_config()
            retriever = WithinDatasetRetrival(config=viz_config)
            # query_box = {'z': 3, 'y_min': 50, 'y_max': 120, 'x_min': 250, 'x_max': 290}
            # query_vec = retriever.get_query_vector(z=3, y=200, x=200, window=3)
            query_vec = retriever.get_query_from_box() 
            sim_map = retriever.compute_heatmap(query_vec)
            retriever.precision_at_k(sim_map)
            title= input("Title of dataset(e.g- pancreas cell chunk 0):")
            retriever.plot_overlay(
                sim_map   = sim_map,
                title     = f'Within-dataset — {title} ',
            )

        except Exception as e:
            print(f"🛑 Error: {e}")
            raise e



if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = VizPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e