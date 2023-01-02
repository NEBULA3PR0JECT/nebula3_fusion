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
        
        # Calculate Intersection
        intersection = bb_intersection(reid_bbox, vc_bbox)
        print("Intersection: {}".format(intersection))

        return intersection

    
    def correct_matches(self, matches):
        """
        Correcting the edge cases of matches between face bbox and its corresponding person bbox
        CASES:
            Two people cases: (They are near each other in the image)
                1. Detected Two face bbox, and One person bboxes.
                2. Detected One face bbox, and two person bboxes.
                3. Detected Two face bboxes, and two person bboxes.
                    - One face bbox intersects with both bboxes, Other face bbox intersects with one bbox.
                    - One face bbox intersects with both bboxes but partially in one of them, Other face bbox intersects with one bbox.
                    - Both faces bboxes intersects with both bboxes.
        """

        # CASE 1: Two (or more) faces intersect with one person bbox
        
        detected_person_bboxes = {}

        corrected_matches = matches.copy()

        for match in matches:
            cur_reid_bbox = str(match['reid_bbox'])
            cur_vc_bbox = str(match['vc_bbox'])
            cur_bboxes_intersection = match['bbox_intersection']
            
            # We detected that the person bbox has already been found.
            if cur_vc_bbox in detected_person_bboxes:
                # Get the previous face intersection
                prev_bboxes_intersection = detected_person_bboxes[cur_vc_bbox]
                
                # Update with the higest intersection (which is the closest face and our hueristic)
                # Then proceed to delete the previous intersection from the matches
                if cur_bboxes_intersection >= prev_bboxes_intersection:
                    detected_person_bboxes.update({cur_vc_bbox: cur_bboxes_intersection})

                    for idx, correct_match in corrected_matches.copy():
                        if correct_match['bbox_intersection'] == prev_bboxes_intersection:
                            del corrected_matches[idx]
            
            # Add the detections.
            if cur_vc_bbox not in detected_person_bboxes:
                detected_person_bboxes.update({cur_vc_bbox: cur_bboxes_intersection})

                
        
        return corrected_matches

        
        

    


def main():
    
    fusion_pipeline = FusionPipeline()
    # movie_ids = ["Movies/7023181708619934815", "Movies/-3873382000557298376", "Movies/5045288714704237341",
    #             "Movies/-1202209992462902069", "Movies/1946038493973736863", "Movies/-7609741451718247625",
    #             "Movies/-638061510228445424", "Movies/-5177664853933870762", "Movies/6959368340271409763",
    #             "Movies/5279939171034674409", "Movies/-7247731179043334982", "Movies/-6432245914174803073"]

    movie_ids = ["Movies/7023181708619934815"]

    for movie_id in movie_ids:
        collection = "s4_re_id"
        reid_detections = fusion_pipeline.get_reid_detections(movie_id = movie_id, collection=collection)
        collection = "s4_visual_clues"
        
        fusion_output = {
            'movie_id': movie_id,
            'frame_numbers': {}
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
            for reid_bbox_obj in reid_bboxes:
                face_id = reid_bbox_obj['id']
                reid_bbox = reid_bbox_obj['bbox']
                # Iterate over Visual Clues bboxes for the specific face. (that have 'person' in them)
                for vc_roi in vc_rois:
                    vc_bbox = vc_roi['bbox']
                    vc_bbox = vc_bbox.replace("[","").replace("]","").split(",")
                    vc_bbox = [float(xy) for xy in vc_bbox]
                    bboxes_intersection = fusion_pipeline.calculate_intersection \
                                            (reid_bbox=reid_bbox, vc_bbox=vc_bbox, movie_id=movie_id, frame_num=reid_frame)
                    
                    # Keep the bounding boxes that have high intersection
                    if bboxes_intersection > INTERSECTION_THRESHOLD:
                        # if str(vc_bbox) not in map_reid_bbox_to_vc_bbox:
                        #     map_reid_bbox_to_vc_bbox[str(vc_bbox)] = ''
                        #     reid_intersections[str(reid_bbox)] = [{
                        #                                     'vc_bbox': vc_bbox,
                        #                                     'bbox_intersection': bboxes_intersection,
                        #                                     'reid_frame': reid_frame
                        #                                     }]
                        # else:
                        #     print("Detected another face for the same person bbox.")
                        #     reid_intersections[str(reid_bbox)].append({
                        #                                     'vc_bbox': vc_bbox,
                        #                                     'bbox_intersection': bboxes_intersection,
                        #                                     'reid_frame': reid_frame
                        #                                     })
                        # Save the information of strong candidates regrding bboxes intersection
                        reid_frame_str = str(reid_frame)
                        if reid_frame_str not in fusion_output['frame_numbers']:
                            fusion_output['frame_numbers'][reid_frame_str] = {'intersections' : []}
                        
                        fusion_output['frame_numbers'][reid_frame_str]['intersections'].append(
                            {
                                'reid_bbox': reid_bbox,
                                'vc_bbox': vc_bbox,
                                'bbox_intersection': bboxes_intersection,
                                'face_id': face_id
                            }
                        )
                            
            # Check BBox Intersection Edge cases in the current frame
            
            # print(reid_intersections)
        print(fusion_output)
        # fusion_pipeline.insert_json_to_db(fusion_output, fusion_pipeline.collection_name)

        save_image = True
        if save_image:
            movie_id = fusion_output['movie_id']
            frames = fusion_output['frame_numbers']
            for frame_num, _ in frames.items():
                matches = []
                # Get all the matches for the current frame
                for intersection in frames[str(frame_num)]['intersections']:
                    matches.append({
                                    'reid_bbox': intersection['reid_bbox'],
                                    'vc_bbox':   intersection['vc_bbox'],
                                    'bbox_intersection': intersection['bbox_intersection'],
                                    'face_id':   intersection['face_id']
                                })
                
                post_processed_matches = fusion_pipeline.correct_matches(matches)
                
                # Draw all the matches on the current frame
                if post_processed_matches:
                    
                    image_url = fusion_pipeline.get_image_url(movie_id, frame_num=int(frame_num), collection="s4_visual_clues")
                    movie_name = image_url.split("/")[-2]
                    save_img_with_bboxes(bbox_details=matches, image_url=image_url, \
                                    frame_num=frame_num, movie_name=movie_name)



if __name__ == '__main__':
    main()
