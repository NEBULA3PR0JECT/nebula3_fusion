import numpy as np
from database.arangodb import DBBase
import cv2
from pathlib import Path
import csv
import requests

import tqdm
from PIL import Image
import time
import random
import os, sys

# from visual_clues.bboxes_implementation import DetectronBBInitter

URL_PREFIX = "http://74.82.29.209:9000"
CUR_FOLDER = os.path.dirname(os.path.abspath(__file__))

class FusionPipeline:
    def __init__(self):
        self.nre = DBBase()
        print("Connected to database: {}".format(self.nre.database))
        self.collection_name = "s4_fusion"


    def insert_json_to_db(self, combined_json, collection_name):
        """
        Inserts a JSON with global & local tokens to the database.
        """

        res = self.nre.write_doc_by_key(combined_json, collection_name, overwrite=True, key_list=['movie_id', 'frame_num'])

        print("Successfully inserted to database. Collection name: {}, movie_id: {}".format(collection_name, combined_json['movie_id']))
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
    
    def plot_one_box(self, x, img, color=None, label=None, line_thickness=3):
        # Plots one bounding box on image img
        tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
        color = color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)

    def save_img_with_bboxes(self, reid_bbox, vc_bbox, image_url, frame_num):
        
        resp = requests.get(image_url, stream=True).raw
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        img_orig = cv2.imdecode(image, cv2.IMREAD_COLOR)
        self.plot_one_box(reid_bbox, img_orig, label=None, color=None, line_thickness=3)
        self.plot_one_box(vc_bbox, img_orig, label=None, color=None, line_thickness=3)
        # TO DELETE - SANITY TEXT OF BBOXES ON IMAGE
        cv2.imwrite(os.path.join(CUR_FOLDER, "images/frame_{}.jpg".format(frame_num)), img_orig)
        print(f"The image is saved with bboxes")
        return
    
    def get_image_url(self, movie_id, frame_num, collection):
        try:
            data = self.nre.get_doc_by_key({'movie_id': movie_id, 'frame_num': frame_num}, collection)
        except KeyError:
            print("Movie ID {} and Fra not found.".format(movie_id))
        return data['url']


    def calculate_iou(self, reid_bbox, vc_bbox, movie_id, frame_num):
        """
        Calculate IOU between REID bbox and Visual Clues bbox.
        """
        save_image = True
        if save_image:
            image_url = self.get_image_url(movie_id, frame_num=frame_num, collection="s4_visual_clues")
            self.save_img_with_bboxes(reid_bbox=reid_bbox, vc_bbox=vc_bbox, image_url=image_url, frame_num=frame_num)
        # Calculate IOU
        
        return
        

    


def main():
    
    fusion_pipeline = FusionPipeline()
    movie_id = "Movies/-5164132544733975037"
    collection = "s4_re_id"
    reid_detections = fusion_pipeline.get_reid_detections(movie_id = movie_id, collection=collection)
    collection = "s4_visual_clues"
    
    # Iterate over all the RE-ID frames.
    for reid_detection in reid_detections:
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
                fusion_pipeline.calculate_iou(reid_bbox=reid_bbox, vc_bbox=vc_bbox, movie_id=movie_id, frame_num=reid_frame)





if __name__ == '__main__':
    main()
