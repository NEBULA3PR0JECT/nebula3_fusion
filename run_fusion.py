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
import ast
from utils.image_utils import bb_intersection_over_union, bb_intersection, \
                                bb_smallest_area, plot_one_box, save_img_with_bboxes, \
                                    bb_hueristic_face_coordinate, bb_center_coordinate, \
                                        distance_between_two_points

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
        if not data:
            return None
        reid_detections = data['frames'] if 'frames' in data else []
        return reid_detections

    def get_reid_face_ids(self, movie_id, frame_num, collection):
        """
        Get the bboxes that have 'person' detected in them.
        """
        reid_data = self.get_reid_detections(movie_id, collection)
        face_ids = set()
        for reid_det in reid_data:
            if str(reid_det['frame_num']) == str(frame_num):
                reid_bboxes = reid_det['re-id']
                for reid_bbox in reid_bboxes:
                    face_ids.add(str(reid_bbox['id']))
        return face_ids
    
    def get_visual_clues_data(self, movie_id, collection, frame_num):
        try:
            data = self.nre.get_doc_by_key({'movie_id': movie_id, 'frame_num': frame_num}, collection)
        except KeyError:
            print("Movie ID {} and Frame not found.".format(movie_id))
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
    
    def get_visual_clues_person_ids(self, movie_id, frame_num, collection):
        """
        Get the bboxes that have 'person' detected in them.
        """
        visual_clue_data = self.get_visual_clues_data(movie_id, collection, frame_num)
        vc_ids = set()
        vc_ids_data = visual_clue_data['roi']
        for vc_roi in vc_ids_data:
            if 'person' in vc_roi['bbox_object']:
                vc_ids.add(vc_roi['roi_id'])
        return vc_ids
    
    
    def get_image_url(self, movie_id, frame_num, collection):
        try:
            data = self.nre.get_doc_by_key({'movie_id': movie_id, 'frame_num': frame_num}, collection)
        except KeyError:
            print("Movie ID {} and Frame num {} not found.".format(movie_id, frame_num))
        return data['url']
    
    def get_movie_ids_by_tag(self, tag, collection):

        results = []
        query = 'FOR doc IN {} RETURN doc'.format(collection)
        cursor = self.nre.db.aql.execute(query)
        for doc in cursor:
            results.append(doc)
        temp_results = []
        for result in results:
            if result['movies']:
                if 'benchmark' in result['inputs']['videoprocessing']:
                    if result['inputs']['videoprocessing']['benchmark']['benchmark_tag'] == 'v100':
                        temp_results.append(list(result['movies'].keys())[0])
        return temp_results


    def calculate_intersection(self, reid_bbox, vc_bbox, movie_id, frame_num):
        """
        Calculate IOU between REID bbox and Visual Clues bbox.
        """
        
        # Calculate Intersection
        intersection = bb_intersection(reid_bbox, vc_bbox)
        print("Intersection: {}".format(intersection))

        return intersection
    
    def calc_iou_on_matches(self, matches):
        """
        Calclate IOU on all matches
        """
        return matches


    
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
        
        detected_person_bboxes, detected_reid_bboxes = {}, {}

        corrected_matches = matches.copy()

        for match in matches:
            cur_reid_bbox = str(match['reid_bbox'])
            cur_vc_bbox = str(match['vc_bbox'])
            cur_face_area = match['face_area']
            
            # We detected that the person bbox has already been found.
            if cur_vc_bbox in detected_person_bboxes:
                # Get the previous face area
                prev_face_area = detected_person_bboxes[cur_vc_bbox]
                
                temp_corrected_matches = corrected_matches.copy()
                # Update with the higest area (which is the closest face and our hueristic)
                # Then proceed to delete the previous intersection from the matches
                if cur_face_area > prev_face_area:
                    detected_person_bboxes.update({cur_vc_bbox: cur_face_area})

                    for idx, correct_match in enumerate(temp_corrected_matches):
                        if correct_match['face_area'] == prev_face_area:
                            del temp_corrected_matches[idx]
                            corrected_matches = temp_corrected_matches
                
                # Delete the current face area which is smaller/equal than the highest face area
                # It's equal when a face is fully intersected with two different person bounding boxes.
                else:
                    for jdx, correct_match in enumerate(temp_corrected_matches):
                        # The "str(correct_match['vc_bbox']) == cur_vc_bbox" is important because we want to make sure
                        # That we delete the face is associated to the same person bbox
                        # Because the same face can be inside multiple person bboxes with same face area
                        # If two people are close to one another their bboxes can be big enough to contain the same face
                        # So we have to make sure the person bbox is the one that we already detected
                        if correct_match['face_area'] == cur_face_area and \
                                str(correct_match['vc_bbox']) == cur_vc_bbox: 
                                
                            del temp_corrected_matches[jdx]
                            corrected_matches = temp_corrected_matches

            
            # Add the detections.
            if cur_vc_bbox not in detected_person_bboxes:
                detected_person_bboxes.update({cur_vc_bbox: cur_face_area})

            
            temp_corrected_matches = corrected_matches.copy()
            # We detected that the face bbox has already been found.
            if cur_reid_bbox in detected_reid_bboxes:
                # Calculate the huristic face ('upper half of bbox') center coordinate of
                # prev face & person bbox and current ones
                prev_vc_bbox = ast.literal_eval(detected_reid_bboxes[cur_reid_bbox])
                cur_reid_bbox = ast.literal_eval(cur_reid_bbox)
                prev_upper_cen_coord = bb_hueristic_face_coordinate(prev_vc_bbox)
                cur_vc_bbox = ast.literal_eval(cur_vc_bbox)
                curr_upper_cen_coord = bb_hueristic_face_coordinate(cur_vc_bbox)

                # Check which person bbox is hueristically better matched to the current face bbox
                # By checking the euclidian distance between the center face and upper center person bboxes.
                face_center_coord = bb_center_coordinate(cur_reid_bbox)

                prev_vc_bbox_distance = distance_between_two_points(prev_upper_cen_coord, face_center_coord)
                curr_vc_bbox_distance = distance_between_two_points(curr_upper_cen_coord, face_center_coord)

                if prev_vc_bbox_distance > curr_vc_bbox_distance:
                    for idx, correct_match in enumerate(temp_corrected_matches):
                        # We saved the person bbox in `detected_reid_bboxes[cur_reid_bbox]`
                        # Thats how we can uniquely idenify the person bbox which is further away
                        # from the current person bbox.
                        if str(correct_match['vc_bbox']) == detected_reid_bboxes[str(cur_reid_bbox)]:
                            del temp_corrected_matches[idx]
                            corrected_matches = temp_corrected_matches
                else:
                    # Delete the current face & person bbox
                    for idx, correct_match in enumerate(temp_corrected_matches):
                        if correct_match['vc_bbox'] == cur_vc_bbox and \
                            correct_match['reid_bbox'] == cur_reid_bbox:
                            del temp_corrected_matches[idx]
                            corrected_matches = temp_corrected_matches
                
            # Add the face detections with their current person bbox for future
            # calculations such as finding the center coordinate
            if str(cur_reid_bbox) not in detected_reid_bboxes:
                detected_reid_bboxes.update({cur_reid_bbox: cur_vc_bbox})
        

        return corrected_matches

        
        

    


def main():
    
    fusion_pipeline = FusionPipeline()
    tag='v100'
    collection='pipelines'
    # movie_ids = fusion_pipeline.get_movie_ids_by_tag(tag, collection)

    # movie_ids = ["Movies/7023181708619934815", "Movies/-3873382000557298376", "Movies/5045288714704237341",
    #             "Movies/-1202209992462902069", "Movies/1946038493973736863", "Movies/-7609741451718247625",
    #             "Movies/-638061510228445424", "Movies/-5177664853933870762", "Movies/6959368340271409763",
    #             "Movies/5279939171034674409", "Movies/-7247731179043334982", "Movies/-6432245914174803073"]

    # movie_ids = fusion_pipeline.get_movie_ids_by_tag(tag='v100', collection='pipelines')

    # Trailers:
    movie_ids = ["Movies/1921717892733313742", "Movies/-7355878014542434114", "Movies/-1932219743953950323",
                "Movies/5718056395198158653", "Movies/5421091196518235613", "Movies/9190480897184314431",
                "Movies/5752769488301225156", "Movies/-8052325165495258532", "Movies/2919871177174099132"]

    # List of faces without person bboxes attached to them (Only REID)
    face_ids = set()
    # List of person without faces attached to them (Only Visual Clues)
    vc_ids = set()

    # Trailers:
    movie_ids = ["Movies/1921717892733313742", "Movies/-7355878014542434114", "Movies/-1932219743953950323",
                "Movies/5718056395198158653", "Movies/5421091196518235613", "Movies/9190480897184314431",
                "Movies/5752769488301225156", "Movies/-8052325165495258532", "Movies/2919871177174099132"]

    # movie_ids = ["Movies/5421091196518235613"]

    skipped_movie_ids = []
    for movie_id in movie_ids:
        print("Working on Movie ID: {}".format(movie_id))
        collection = "s4_re_id"
        reid_detections = fusion_pipeline.get_reid_detections(movie_id = movie_id, collection=collection)
        if not reid_detections:
            print("Skipping Movie ID: {}, Because REID detections were not found".format(movie_id))
            skipped_movie_ids.append(movie_id)
            continue
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
            if not vc_data:
                 print("Skipping Movie ID: {}, Because vc_data was not found".format(movie_id))
                 skipped_movie_ids.append(movie_id)
                 continue
            vc_rois = fusion_pipeline.get_visual_clues_rois(visual_clue_data=vc_data)
            if not vc_rois:
                print("Skipping Movie ID: {}, Because vc_rois was not found".format(movie_id))
                skipped_movie_ids.append(movie_id)
                continue
                
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
                        # Save the information of strong candidates regrding bboxes intersection
                        reid_frame_str = str(reid_frame)
                        if reid_frame_str not in fusion_output['frame_numbers']:
                            fusion_output['frame_numbers'][reid_frame_str] = {'intersections' : [], 'ious': []}
                        
                        face_area = bb_smallest_area(reid_bbox, vc_bbox)
                        iou = bb_intersection_over_union(reid_bbox, vc_bbox)
                        fusion_output['frame_numbers'][reid_frame_str]['intersections'].append(
                            {
                                'reid_bbox': reid_bbox,
                                'vc_bbox': vc_bbox,
                                'bbox_intersection': bboxes_intersection,
                                'face_area': face_area,
                                'face_id': str(face_id),
                                'iou': iou,
                                'vc_id': str(vc_roi['roi_id'])
                            }
                        )
                        if bboxes_intersection == 1.0:
                            fusion_output['frame_numbers'][reid_frame_str]['ious'].append(
                                {
                                    'iou': bboxes_intersection,
                                    'face_id': str(face_id),
                                    'vc_id': str(vc_roi['roi_id'])
                                }
                            )
                        
            # print(reid_intersections)
        # print(fusion_output)
        # fusion_pipeline.insert_json_to_db(fusion_output, fusion_pipeline.collection_name)

        save_image = True
        if save_image:
            movie_id = fusion_output['movie_id']
            frames = fusion_output['frame_numbers']
            for frame_num, _ in frames.items():
                
                image_url = fusion_pipeline.get_image_url(movie_id, frame_num=int(frame_num), collection="s4_visual_clues")
                movie_name = image_url.split("/")[-2]
                print("Working on movie: {}, frame: {}".format(movie_name, frame_num))

                matches = []
                # Get all the matches for the current frame
                for intersection in frames[str(frame_num)]['intersections']:
                    matches.append({
                                    'reid_bbox':            intersection['reid_bbox'],
                                    'vc_bbox':              intersection['vc_bbox'],
                                    'bbox_intersection':    intersection['bbox_intersection'],
                                    'face_area':            intersection['face_area'],
                                    'face_id':              intersection['face_id'],
                                    'vc_id':                intersection['vc_id']
                                })
                
                post_processed_matches = fusion_pipeline.correct_matches(matches)

                vc_ids = fusion_pipeline.get_visual_clues_person_ids(movie_id, int(frame_num), collection="s4_visual_clues")
                face_ids = fusion_pipeline.get_reid_face_ids(movie_id, frame_num, collection="s4_re_id")
                for idx, post_processed_match in enumerate(post_processed_matches):
                        face_id = str(post_processed_match['face_id'])
                        vc_id = str(post_processed_match['vc_id'])

                        if face_id in face_ids:
                            face_ids.remove(face_id)
                        if vc_id in vc_ids:
                            vc_ids.remove(vc_id)
                        

                for iou_data in frames[str(frame_num)]['ious']:
                    print("-------- IOUs Frame Number: {} -------".format(str(frame_num)))
                    print("VC_ID: {}".format(iou_data['vc_id']))
                    print("FACE_ID: {}".format(iou_data['face_id']))
                    print("IOU: {}".format(iou_data['iou']))
                    print("-"*20)
                # Draw all the matches on the current frame

                #faces_no_person.append({'roi_id': vc_roi['roi_id']})
                #person_no_faces.append(v)
                if post_processed_matches:
                    
                    save_img_with_bboxes(bbox_details=post_processed_matches, image_url=image_url, \
                                    frame_num=frame_num, movie_name=movie_name)
    print("Skipped movie ids: {}".format(skipped_movie_ids))


if __name__ == '__main__':
    main()
