import numpy as np
from database.arangodb import DBBase
import cv2
from pathlib import Path
import csv

import tqdm
from PIL import Image
import time
import random
import os, sys

from utils.image_utils import bb_intersection_over_union, bb_intersection, \
                                        plot_one_box, save_img_with_bboxes

# from visual_clues.bboxes_implementation import DetectronBBInitter

URL_PREFIX = "http://74.82.29.209:9000"
CUR_FOLDER = os.path.dirname(os.path.abspath(__file__))
INTERSECTION_THRESHOLD = 0.97

class FusionPipeline:
    def __init__(self):
        self.nre = DBBase()
        print("Connected to database: {}".format(self.nre.database))
        self.collection_name = "s4_fusion"


    def insert_json_to_db(self, json_obj, collection_name):
        """
        Inserts a JSON with global & local tokens to the database.
        """

        res = self.nre.write_doc_by_key(json_obj, collection_name, overwrite=True, key_list=['movie_id'])

        print("Successfully inserted to database. Collection name: {}, movie_id: {}".format(collection_name, json_obj['movie_id']))
        return res

    def get_mdf_urls_from_db(self, movie_id, collection):

        data = self.nre.get_doc_by_key({'_id': movie_id}, collection)
        urls = []
        if not data:
            print("{} not found in database. ".format(movie_id))
            return False
        if 'mdfs_path' not in data:
            print("MDFs cannot be found in {}".format(movie_id))
            return False
        for mdf_path in data['mdfs_path']:
            url = os.path.join(URL_PREFIX, mdf_path[1:])
            urls.append(url)
        return urls
    
    def get_pipelineid_from_db(self, movie_id, collection):

        data = self.nre.get_doc_by_key({'_id': movie_id}, collection)
        if not data:
            print("{} not found in database. ".format(movie_id))
            return False
        if 'pipeline_id' not in data:
            print("pipeline_id cannot be found in {}".format(movie_id))
            return False
        pipeline_id = data['pipeline_id']

        return pipeline_id

    def get_input_type_from_db(self, pipeline_id, collection):

        pipeline_data = self.nre.get_doc_by_key({'_key': pipeline_id}, collection)
        if pipeline_data:
            if "dataset" in pipeline_data["inputs"]["videoprocessing"]:
                input_type = pipeline_data["inputs"]["videoprocessing"]["dataset"]["type"]
            else:
                input_type = pipeline_data["inputs"]["videoprocessing"]["movies"][0]["type"]
        return input_type
    
        
    def run_fusion_pipeline(self, movie_id):
        print("Starting to record time of visual clues!")
        start_time = time.time()



        end_time = time.time() - start_time
        print("Total time it took for visual clues: {}".format(end_time))
        return True, None

    def get_reid_detections(self, movie_id, collection):
        try:
            data = self.nre.get_doc_by_key({'movie_id': movie_id}, collection)
        except KeyError:
            print("Movie ID {} not found.".format(movie_id))
        reid_detections = data['frames'] if 'frames' in data else []
        return reid_detections
    
    def get_visual_clues_data(self, movie_id, collection, frame_num):
        try:
            data = self.nre.get_doc_by_key({'movie_id': movie_id, 'frame_num': frame_num}, collection)
        except KeyError:
            print("Movie ID {} and Fra not found.".format(movie_id))
        return data

    
    def get_visual_clues_rois(self, visual_clue_data):
        """
        Get the bboxes that have 'person' detected in them.
        """
        vc_rois = []
        vc_rois_data = visual_clue_data['roi']
        for vc_roi in vc_rois_data:
            if 'person' in vc_roi['bbox_object']:
                vc_rois.append(vc_roi)
        return vc_rois
    
    
    def get_image_url(self, movie_id, frame_num, collection):
        try:
            data = self.nre.get_doc_by_key({'movie_id': movie_id, 'frame_num': frame_num}, collection)
        except KeyError:
            print("Movie ID {} and Fra not found.".format(movie_id))
        return data['url']


    def calculate_intersection(self, reid_bbox, vc_bbox, movie_id, frame_num):
        """
        Calculate IOU between REID bbox and Visual Clues bbox.
        """
        save_image = True
        
        # Calculate Intersection
        intersection = bb_intersection(reid_bbox, vc_bbox)
        print("Intersection: {}".format(intersection))

        if intersection > INTERSECTION_THRESHOLD:
            if save_image:
                image_url = self.get_image_url(movie_id, frame_num=frame_num, collection="s4_visual_clues")
                save_img_with_bboxes(reid_bbox=reid_bbox, vc_bbox=vc_bbox, image_url=image_url, frame_num=frame_num)
        return intersection
        

    


def main():
    
    fusion_pipeline = FusionPipeline()
    movie_id = "Movies/-5164132544733975037"
    collection = "s4_re_id"
    reid_detections = fusion_pipeline.get_reid_detections(movie_id = movie_id, collection=collection)
    collection = "s4_visual_clues"
    
    fusion_output = {
        'movie_id': movie_id,
        'frames': []
    }

    # Iterate over all the RE-ID frames.
    for reid_detection in reid_detections:

        # Mapping between RE-ID details (bbox, frame_num, intersection confidence) and VC bboxes
        reid_intersections = {}
        # Mapping between RE-ID bboxes to VC bboxes to save which ones have high intersection
        map_reid_bbox_to_vc_bbox = {}

        reid_frame = reid_detection['frame_num']
        vc_data = fusion_pipeline.get_visual_clues_data(movie_id = movie_id, collection=collection, frame_num=reid_frame)
        vc_rois = fusion_pipeline.get_visual_clues_rois(visual_clue_data=vc_data)
        
        reid_bboxes = reid_detection['re-id']
        # Iterate over the RE-ID face(s) in the current frame (There may be multiple different Face IDs)
        for reid_bbox in reid_bboxes:
            reid_bbox = reid_bbox['bbox']
            # Iterate over Visual Clues bboxes for the specific face. (that have 'person' in them)
            for vc_roi in vc_rois:
                vc_bbox = vc_roi['bbox']
                vc_bbox = vc_bbox.replace("[","").replace("]","").split(",")
                vc_bbox = [float(xy) for xy in vc_bbox]
                bboxes_intersection = fusion_pipeline.calculate_intersection \
                                        (reid_bbox=reid_bbox, vc_bbox=vc_bbox, movie_id=movie_id, frame_num=reid_frame)
                
                # Keep the bounding boxes that have high intersection
                if bboxes_intersection > INTERSECTION_THRESHOLD:
                    if str(vc_bbox) not in map_reid_bbox_to_vc_bbox:
                        map_reid_bbox_to_vc_bbox[str(vc_bbox)] = ''
                        reid_intersections[str(reid_bbox)] = [{
                                                        'vc_bbox': vc_bbox,
                                                        'bbox_intersection': bboxes_intersection,
                                                        'reid_frame': reid_frame
                                                        }]
                    else:
                        print("Detected another face for the same person bbox.")
                        reid_intersections[str(reid_bbox)].append({
                                                        'vc_bbox': vc_bbox,
                                                        'bbox_intersection': bboxes_intersection,
                                                        'reid_frame': reid_frame
                                                        })
                    # Save the information of strong candidates regrding bboxes intersection
                    fusion_output['frames'].append({
                                            'frame_num': reid_frame,
                                            'reid_bbox': reid_bbox,
                                            'vc_bbox': vc_bbox,
                                            'bbox_intersection': bboxes_intersection,
                                        })
        # Check BBox Intersection Edge cases in the current frame
          
        print(reid_intersections)
        debug_p = 0
    print(fusion_output)
    fusion_pipeline.insert_json_to_db(fusion_output, fusion_pipeline.collection_name)
    a=0





if __name__ == '__main__':
    main()
