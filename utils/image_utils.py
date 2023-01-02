import requests
import numpy as np
import cv2
import random
import os
import PIL.ImageColor as ImageColor
CUR_FOLDER = os.path.abspath(__file__ + "/../../")

def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou

def bb_intersection(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    smallBboxArea = min(boxAArea, boxBArea)
    if smallBboxArea <= interArea and interArea > 0:
        return smallBboxArea / interArea
    if smallBboxArea > interArea and smallBboxArea > 0:
        return interArea / smallBboxArea

    return 0

def bb_smallest_area(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    smallBboxArea = min(boxAArea, boxBArea)

    return smallBboxArea

def bb_center_coordinate(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    cX = (xA + xB) / 2
    cY = (yA + yB) / 2

    return (cX, cY)

def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    cv2.putText(img, str(label), (c2[0], c2[1]), 0, tl / 3, [225, 255, 255], thickness=max(tl - 1, 1), lineType=cv2.LINE_AA)

def save_img_with_bboxes(bbox_details, image_url, frame_num, movie_name):
    
    resp = requests.get(image_url, stream=True).raw
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    img_orig = cv2.imdecode(image, cv2.IMREAD_COLOR)
    color_space = [ImageColor.getrgb(n) for n, c in ImageColor.colormap.items()][7:]
    for cur_details in bbox_details:
        reid_bbox = cur_details['reid_bbox']
        vc_bbox = cur_details['vc_bbox']
        face_id = cur_details['face_id']
        color = [random.randint(0, 255) for _ in range(3)]
        plot_one_box(reid_bbox, img_orig, label=face_id, color=color, line_thickness=3)
        plot_one_box(vc_bbox, img_orig, label=face_id, color=color, line_thickness=3)
    # TO DELETE - SANITY TEXT OF BBOXES ON IMAGE
    movie_name_path =  os.path.join(CUR_FOLDER, "images/{}".format(movie_name))
    if not os.path.exists(movie_name_path):
        os.makedirs(movie_name_path)
    cv2.imwrite(os.path.join(movie_name_path, "frame_{}.jpg".format(frame_num)), img_orig)
    print(f"The image is saved with bboxes")
    return