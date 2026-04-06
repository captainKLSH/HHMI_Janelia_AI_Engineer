from src.task2 import logger
import argparse
import os
import sys

from src.task2.pipeline.stage1 import DataIngestionTrainingPipeline
from src.task2.pipeline.stage2 import ModelBuildingPipeline
from src.task2.pipeline.stage3 import VizPipeline


def main():
    parser = argparse.ArgumentParser(description="Run specific stages of the ML Pipeline.")

    parser.add_argument('-all', action='store_true', help="Run the entire pipeline")
    parser.add_argument('-stage1', action='store_true', help="Run Data Ingestion")
    parser.add_argument('-stage2', action='store_true', help="Run Model Build" )
    parser.add_argument('-stage3', action='store_true', help="Within Data Retrival" )

    args = parser.parse_args()


    
    run_all = args.all or not any([
        args.stage1, args.stage2, args.stage3
    ])

    

    # --- PIPELINE STAGES ---

    if run_all or args.stage1:
        STAGE_NAME = "Data Ingestion stage"
        try:
            logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
            obj = DataIngestionTrainingPipeline()
            obj.main()
            logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
        except Exception as e:
            logger.exception(e)
            raise e
        
    if run_all or args.stage2: 
        STAGE_NAME = "Model Building stage"

        try:
            logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
            obj = ModelBuildingPipeline()
            obj.main()
            logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
        except Exception as e:
            logger.exception(e)
            raise e
        
    if run_all or args.stage3: 
        STAGE_NAME = "Within data Visualization Retrival stage"

        try:
            logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
            obj = VizPipeline()
            obj.main()
            logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
        except Exception as e:
            logger.exception(e)
            raise e



if __name__ == "__main__":
    main()