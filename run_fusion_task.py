from experts.pipeline.api import PipelineApi, PipelineTask
from fusion_task import FusionPipeline
import os
from typing import Tuple

def test_pipeline_task(pipeline_id):
    class MyTask(PipelineTask):
        def __init__(self):
            self.fusion_pipeline = FusionPipeline()
            print("Initialized successfully.")

        def process_movie(self, movie_id: str) -> Tuple[bool, str]:
            print (f'handling movie: {movie_id}')

            output = self.fusion_pipeline.run_fusion_pipeline(movie_id)

            print("Finished handling movie.")
            print(output)
            return output
        def get_name(self) -> str:
            return "fusion"

    pipeline = PipelineApi(None)
    task = MyTask()
    pipeline.handle_pipeline_task(task, pipeline_id, stop_on_failure=True)

def test():
    pipeline_id = os.environ.get('PIPELINE_ID')
    # pipeline_id = "dff9a316-4296-45b1-a503-359bb8457bba" 
    # print(pipeline_id)
    # pipeline_id='12345678'
    test_pipeline_task(pipeline_id)

if __name__ == '__main__':
    test()
