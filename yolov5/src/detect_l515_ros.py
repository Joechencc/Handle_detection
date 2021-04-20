#!/usr/bin/python3

import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import rospy
#from message_filters import ApproximateTimeSynchronizer, Subscriber
import message_filters
from sensor_msgs.msg import CompressedImage, Image

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, non_max_suppression, apply_classifier, scale_coords, \
    xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
import numpy as np
from cv_bridge import CvBridge
import math
#from yolov5.msg import DoorLocationPacket

class rigid_body:
    lu_x = 0
    lu_y = 0
    ru_x = 0
    ru_y = 0
    ld_x = 0
    ld_y = 0
    rd_x = 0
    rd_y = 0

class door_body:
    HT_x = 0
    HT_y = 0
    OT_x = 0
    OT_y = 0
    HB_x = 0
    HB_y = 0
    OB_x = 0
    OB_y = 0


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh) 
    
def door_plane(img, door_flap):
    global handle_position
    door_plane_right, door_plane_left = 0,0
    w_min, w_max, h_min, h_max = 0,0,0,0

    # print("door_flap.HB_x:::::"+str(door_flap.HB_x))
    # print("door_flap.HT_x:::::"+str(door_flap.HT_x))
    # print("door_flap.OT_x:::.::"+str(door_flap.OT_x))
    # print("door_flap.OB_x:::::"+str(door_flap.OB_x))

    # print("door_flap.HB_y:::::"+str(door_flap.HB_y))
    # print("door_flap.HT_y:::::"+str(door_flap.HT_y))
    # print("door_flap.OT_y:::.::"+str(door_flap.OT_y))
    # print("door_flap.OB_y:::::"+str(door_flap.OB_y))
    if (not (door_flap.OT_x and door_flap.OB_x and door_flap.HT_x and door_flap.HB_x)):
        return None, None, None, None, False

    if handle_position == "left":          
        w_min = max(door_flap.OT_x, door_flap.OB_x)
        w_max = min(door_flap.HT_x, door_flap.HB_x)
        h_min = max(door_flap.OT_y, door_flap.HT_y)
        h_max = min(door_flap.OB_y, door_flap.HB_y)
    else:
        w_min = max(door_flap.HT_x, door_flap.HB_x)
        w_max = min(door_flap.OT_x, door_flap.OB_x)
        h_min = max(door_flap.HT_y, door_flap.OT_y)
        h_max = min(door_flap.HB_y, door_flap.OB_y)
    #print("img::::::::"+str(img.shape))
    # print("w_min::::::::"+str(w_min))
    # print("w_max::::::::"+str(w_max))
    # print("h_min::::::::"+str(h_min))
    # print("h_max::::::::"+str(h_max))

    # xyxy = [0,0,0,0]
    # (xyxy[1]),(xyxy[3]) = h_min,h_max
    # (xyxy[0]),(xyxy[2]) = w_min,w_max
    # plot_one_box(xyxy, img, label="label", color=[0,0,0], line_thickness=3)

    # cv2.imshow("door flap", img)
    
    # cv2.waitKey(3000)

    door_img = img[h_min:h_max,w_min:w_max]
    height, width = h_max - h_min, w_max - w_min

    sample_number_height, height_param, height_start = 10, 0.6, 20
    height_step = math.floor(height_param* height / sample_number_height)
    # print("height_step:::::::"+str(height_step))
    height_end = height_start+height_step* sample_number_height -1


    sample_number_width, width_param, width_start = 10, 0.8, 10
    width_step = math.floor(width_param* width / sample_number_width)
    # print("width_step:::::::"+str(width_step))
    if width_step == 0:
        return None, None, True, True
    else:
        orthogonal_flag = False

    width_end = width_start+width_step* sample_number_width -1

    #A_matrix

    height_array = np.array([x for x in range(height_start, height_end, height_step) for y in range(width_start, width_end, width_step)])
    width_array = np.array([x for y in range(width_start, width_end, width_step) for x in range(height_start, height_end, height_step)])
    depth_array = door_img[height_start:height_end:height_step, width_start:width_end:width_step].flatten()
    ones_array = np.ones_like(depth_array)

    #print("height_array:::::::::::::::::::::::"+str(height_array.shape))
    #print("width_array:::::::::::::::::::::::"+str(width_array.shape))
    #print("depth_array:::::::::::::::::::::::"+str(depth_array.shape))
    if len(height_array) == len(width_array) == len(depth_array):
        integrity_flag = True
    else:
        integrity_flag = False
        return None, None, False, False

    A_matrix = np.vstack((height_array, width_array,depth_array, ones_array)).T
    _, s, vh = np.linalg.svd(A_matrix, full_matrices = False)
    min_idx = np.argmin(s)
    min_vh = vh[:,min_idx]
    n_vector = min_vh[:3]
    vh_norm =  n_vector / np.linalg.norm(n_vector)
    i = 0

    # print("door_img:::::::::"+str(door_img))
    
    # while(door_plane_left == 0):
    #     print("i:::"+str(i))
    #     door_plane_left = door_img[int(height_start)+i , width_start+1]
    #     i = i + 5
    # i = 0
    
    # while((door_img[int(height_end - i), width_end-2] != 0) and (height_end - i)>0):
    #     door_plane_right = door_img[int(height_end - i), width_end-1]
    #     i = i + 5
    
    # print("door_plane_left:::::::"+str(door_plane_left))
    # print("door_plane_right:::::::"+str(door_plane_right))
    return vh_norm, push_pull_state, integrity_flag, orthogonal_flag

def frame_plane(img, door_frame):
    #print("xyxy:::::::::"+str(xyxy))
    global handle_position
    lu_x, lu_y, ld_x, ld_y, ru_x, ru_y, rd_x, rd_y = 0,0,0,0,0,0,0,0
    if (not (door_frame.OT_x and door_frame.OB_x and door_frame.HT_x and door_frame.HB_x)):
        return None, None, None, False
    if handle_position == "left":          
        lu_x = door_frame.OT_x
        lu_y = door_frame.OT_y
        ld_x = door_frame.OB_x
        ld_y = door_frame.OB_y
        ru_x = door_frame.HT_x
        ru_y = door_frame.HT_y
        rd_x = door_frame.HB_x
        rd_y = door_frame.HB_y
    else:
        lu_x = door_frame.HT_x
        lu_y = door_frame.HT_y
        ld_x = door_frame.HB_x
        ld_y = door_frame.HB_y
        ru_x = door_frame.OT_x
        ru_y = door_frame.OT_y
        rd_x = door_frame.OB_x
        rd_y = door_frame.OB_y
        
    # print("lu_x::::"+str(lu_x))
    # print("lu_y::::"+str(lu_y))
    # print("ld_x::::"+str(ld_x))
    # print("ld_y::::"+str(ld_y))
    # print("ru_x::::"+str(ru_x))
    # print("ru_y::::"+str(ru_y))
    # print("rd_x::::"+str(rd_x))
    # print("rd_y::::"+str(rd_y))

    ######### left sampling #########
    sample_number_left, left_param, left_start_x, left_start_y = 10, 1, ld_x, ld_y
    left_delta_x = lu_x - ld_x
    left_delta_y = lu_y - ld_y

    left_x_step = math.floor(left_param* left_delta_x / sample_number_left)
    left_y_step = math.floor(left_param* left_delta_y / sample_number_left)

    left_end_x = left_start_x+left_x_step* sample_number_left -1
    left_end_y = left_start_y+left_y_step* sample_number_left -1

    ######### right sampling #########
    sample_number_right, right_param, right_start_x, right_start_y = 10, 1, rd_x, rd_y

    right_delta_x = ru_x - rd_x
    right_delta_y = ru_y - rd_y

    right_x_step = math.floor(right_param* right_delta_x / sample_number_right)
    right_y_step = math.floor(right_param* right_delta_y / sample_number_right)

    right_end_x = right_start_x+right_x_step* sample_number_right -1
    right_end_y = right_start_y+right_y_step* sample_number_right -1

    #################################

    # A_matrix
    try:
        left_x_array = np.array([x for x in range(left_start_x, left_end_x, left_x_step)])
    except ValueError:
        left_x_array = np.ones((sample_number_left+1,),dtype =int) * int(left_start_x)
    try:
        left_y_array = np.array([y for y in range(left_start_y, left_end_y, left_y_step)])
    except ValueError:
        left_y_array = np.ones((sample_number_left+1,),dtype =int) * int(left_start_y)

    try:
        right_x_array = np.array([x for x in range(right_start_x, right_end_x, right_x_step)])
    except:
        right_x_array = np.ones((sample_number_right+1,),dtype =int) * int(right_start_x)
    try:
        right_y_array = np.array([y for y in range(right_start_y, right_end_y, right_y_step)])
    except:
        right_y_array = np.ones((sample_number_right+1,),dtype =int) * int(right_start_y)

    # Depth Estimate
    depth_left = img[left_y_array, left_x_array]
    depth_right = img[right_y_array, right_x_array]

    ones_array_left = np.ones_like(depth_left)
    ones_array_right = np.ones_like(depth_right)

    assert len(left_x_array) == len(left_y_array) == len(depth_left)
    assert len(right_x_array) == len(right_y_array) == len(depth_right)

    integrity_flag = True

    # Stack A matrix
    A_matrix_left = np.vstack((left_y_array, left_x_array, depth_left, ones_array_left)).T
    A_matrix_right = np.vstack((right_y_array, right_x_array, depth_right, ones_array_right)).T
    A_matrix = np.hstack((A_matrix_left, A_matrix_right))

    _, s, vh = np.linalg.svd(A_matrix, full_matrices = False)
    min_idx = np.argmin(s)
    min_vh = vh[:,min_idx]
    n_vector = min_vh[:3]
    vh_norm =  n_vector / np.linalg.norm(n_vector)
    return img[int((left_start_y+left_end_y)/2), left_start_x], img[int((right_start_y+right_end_y)/2), right_start_x], vh_norm, integrity_flag

    
def detect(image, depth):
    global num_door, num_hinge, num_handle, hinge_array, handle_array, handle_position, push_pull_state
    global door_flap, door_frame
    #cv_image = CvBridge().imgmsg_to_cv2(image, desired_encoding='bgr8')
    #print("image.shape:::::::::::::::::::::::::::::::::::::::"+str(np.array(image_np).shape))
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    
    view_img = True
    save_img = False	 	
    
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    cudnn.benchmark = True  # set True to speed up constant image size inference
    # print("source:::::"+str(source))
      
    #color_arr = np.frombuffer(image.data, np.uint8)
    #image_np = cv2.imdecode(color_arr, cv2.IMREAD_COLOR)
    #im0s = np.expand_dims(image_np, axis=0)
    color_arr = CvBridge().imgmsg_to_cv2(image, desired_encoding='bgr8')
    im0s = np.expand_dims(color_arr, axis=0)
    img0 = im0s.copy()
    
    depth_arr = CvBridge().imgmsg_to_cv2(depth, desired_encoding='16UC1')
    #depth_arr = np.frombuffer(depth.data, np.uint8)
    #print("depth_arr::::::::::"+str(np.asarray(depth_arr)))
    #depth_np = cv2.imdecode(depth_arr, cv2.IMREAD_GRAYSCALE)

    im1s = np.expand_dims(depth_arr, axis=0) # Depth Frame
    img1 = im1s.copy()
    
    # Letterbox
    s = np.stack([letterbox(x, new_shape=imgsz)[0].shape for x in im0s], 0)  # inference shapes
    rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
    im0 = [letterbox(x, new_shape=imgsz, auto=rect)[0] for x in im0s]
    im0 = np.stack(im0, 0)
    im0 = im0[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
        #print("im0:::::::"+str(np.ascontiguousarray(im0).shape))
    im0 = np.ascontiguousarray(im0)
    path = ['4']
        
    
    #for path, img, im0s, vid_cap in dataset_color:
        #print("dataset_depth:::::::::::::;;"+str((dataset_depth)))
        #_,_,im1s,_ = dataset_depth
        #print("img::::::"+str(img.shape))
        #print("im0s::::::"+str(im0s.shape))
    img = torch.from_numpy(im0).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

        # Inference
    t1 = time_synchronized()
    pred = model(img, augment=opt.augment)[0]

        # Apply NMS
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
    t2 = time_synchronized()

    # Apply Classifier
    if classify:
        pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
    for i, det in enumerate(pred):  # detections per image
        #if webcam:  # batch_size >= 1
        s, im0 = '%g: ' % i, im0s[i].copy()
        im1ss = im1s.copy()
        #print("im1:::::::::::"+str(im1.shape))
        #im1 = cv2.convertScaleAbs(im1, alpha=0.03)
        #else:
        #   p, s, im0, frame = path, '', im0s, getattr(dataset_color, 'frame', 0)

        #p = Path(p)  # to Path
        #save_path = str(save_dir / p.name)  # img.jpg
        #txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset_color.mode == 'image' else f'_{frame}')  # img.txt
        s += '%gx%g ' % img.shape[2:]  # print string
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, -1].unique(): 
                n = (det[:, -1] == c).sum()  # detections per class
                s += f'{n} {names[int(c)]}s, '  # add to string
                if names[int(c)] == "door_flap":
                    num_door = int(n)
                elif names[int(c)] == "hinge":
                    num_hinge = int(n)
                    if num_hinge != 0:
                        push_pull_state = "pull"
                        push_sign = -1
                    else:
                        push_pull_state = "push"
                        push_sign = 1
                elif names[int(c)] == "handle":
                    num_handle = int(n)

            # Write results
            for *xyxy, conf, cls in reversed(det):
                if save_txt:  # Write to file
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                    with open(txt_path + '.txt', 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

                if save_img or view_img:  # Add bbox to image
                    label = f'{names[int(cls)]} {conf:.2f}'
                    # print("names[int(cls)]:::"+str(names[int(cls)]))
                    door_flap_check = rigid_body()
                    #assert num_door == 1
                    if (names[int(cls)] == "door_flap"):
                        h_min,h_max = int(xyxy[1]),int(xyxy[3])
                        w_min,w_max = int(xyxy[0]),int(xyxy[2])
                        door_flap_check.lu_x = w_min
                        door_flap_check.lu_y = h_min
                        door_flap_check.ru_x = w_max
                        door_flap_check.ru_y = h_min
                        door_flap_check.ld_x = w_min
                        door_flap_check.ld_y = h_max
                        door_flap_check.rd_x = w_max
                        door_flap_check.rd_y = h_max
                    elif (names[int(cls)] == "hinge"):
                        hinge_body = rigid_body()
                        h_min,h_max = int(xyxy[1]),int(xyxy[3])
                        w_min,w_max = int(xyxy[0]),int(xyxy[2])
                        hinge_body.lu_x = w_min
                        hinge_body.lu_y = h_min
                        hinge_body.ru_x = w_max
                        hinge_body.ru_y = h_min
                        hinge_body.ld_x = w_min
                        hinge_body.ld_y = h_max
                        hinge_body.rd_x = w_max
                        hinge_body.rd_y = h_max
                        hinge_array = [hinge_array, hinge_body]
                        if handle_position == "unknown":
                            if ((hinge_body.lu_x + hinge_body.ru_x) > (door_flap_check.lu_x + door_flap_check.ru_x)):
                                handle_position = "left"
                            else:
                                handle_position = "right"
                    elif (names[int(cls)] == "handle"):
                        handle_body = rigid_body()
                        h_min,h_max = int(xyxy[1]),int(xyxy[3])
                        w_min,w_max = int(xyxy[0]),int(xyxy[2])
                        handle_body.lu_x = w_min
                        handle_body.lu_y = h_min
                        handle_body.ru_x = w_max
                        handle_body.ru_y = h_min
                        handle_body.ld_x = w_min
                        handle_body.ld_y = h_max
                        handle_body.rd_x = w_max
                        handle_body.rd_y = h_max
                        handle_array = [handle_array, handle_body]
                        if handle_position == "unknown":
                            if ((handle_body.lu_x + handle_body.ru_x) < (door_flap_check.lu_x + door_flap_check.ru_x)):
                                handle_position = "left"
                            else:
                                handle_position = "right"

                    elif (names[int(cls)] == "frameHB"):
                        h_min,h_max = float(xyxy[1]),int(xyxy[3])
                        w_min,w_max = float(xyxy[0]),int(xyxy[2])
                        door_frame.HB_x = int((w_min+w_max)/2)
                        door_frame.HB_y = int((h_min+h_max)/2)
                    elif (names[int(cls)] == "frameHT"):
                        h_min,h_max = float(xyxy[1]),int(xyxy[3])
                        w_min,w_max = float(xyxy[0]),int(xyxy[2])
                        # print("w_min,w_max"+str(w_min)+","+str(w_max))
                        door_frame.HT_x = int((w_min+w_max)/2)
                        door_frame.HT_y = int((h_min+h_max)/2)
                        # print("HT_x::::::::"+str(door_frame.HT_x))
                    elif (names[int(cls)] == "frameOB"):
                        h_min,h_max = float(xyxy[1]),int(xyxy[3])
                        w_min,w_max = float(xyxy[0]),int(xyxy[2])
                        door_frame.OB_x = int((w_min+w_max)/2)
                        door_frame.OB_y = int((h_min+h_max)/2)
                    elif (names[int(cls)] == "frameOT"):
                        h_min,h_max = float(xyxy[1]),int(xyxy[3])
                        w_min,w_max = float(xyxy[0]),int(xyxy[2])
                        door_frame.OT_x = int((w_min+w_max)/2)
                        door_frame.OT_y = int((h_min+h_max)/2)
                    elif (names[int(cls)] == "doorHB"):
                        h_min,h_max = float(xyxy[1]),int(xyxy[3])
                        w_min,w_max = float(xyxy[0]),int(xyxy[2])
                        door_flap.HB_x = int((w_min+w_max)/2)
                        door_flap.HB_y = int((h_min+h_max)/2)
                    elif (names[int(cls)] == "doorHT"):
                        h_min,h_max = float(xyxy[1]),int(xyxy[3])
                        w_min,w_max = float(xyxy[0]),int(xyxy[2])
                        door_flap.HT_x = int((w_min+w_max)/2)
                        door_flap.HT_y = int((h_min+h_max)/2)
                    elif (names[int(cls)] == "doorOB"):
                        h_min,h_max = float(xyxy[1]),int(xyxy[3])
                        w_min,w_max = float(xyxy[0]),int(xyxy[2])
                        door_flap.OB_x = int((w_min+w_max)/2)
                        door_flap.OB_y = int((h_min+h_max)/2)
                    elif (names[int(cls)] == "doorOT"):
                        h_min,h_max = float(xyxy[1]),int(xyxy[3])
                        w_min,w_max = float(xyxy[0]),int(xyxy[2])
                        door_flap.OT_x = int((w_min+w_max)/2)
                        door_flap.OT_y = int((h_min+h_max)/2)
                    """
                    if (names[int(cls)] == "door"):
                        print("door detected!!!")
                        print("xyxy::::::::::::::"+str(xyxy))
                        pl_left_depth, pl_right_depth, door_pl, data_integrity = door_plane(im1ss[i], xyxy)
                        #print("pl_left_depth::::::::"+str(pl_left_depth))
                        if data_integrity == False:
                            continue
                        global hinge_position
                        print("hinge_position:::::::::::::"+str(hinge_position))
                        #print("pl_left_depth::::::::::::::"+str(pl_left_depth))
                        #print("pl_right_depth::::::::::::::"+str(pl_right_depth))
                        if hinge_position == "left":
                            h_min,h_max = xyxy[1], xyxy[3]
                            w_min,w_max = xyxy[0], xyxy[2]
                            h_min -= 20
                            if h_min < 0: # do not go beyond the image
                                h_min = 10
                            #h_max += 40
                            w_min -= 20  # do notdoor_plane_right go beyond the image
                            if w_min < 0:
                                w_min = 10
                            w_max += 50
                            xyxy[1], xyxy[3] = h_min,h_max
                            xyxy[0], xyxy[2] = w_min,w_max
                            fr_left_depth, fr_right_depth, frame_pl = frame_plane(im1ss[i], xyxy)
                   #        if w_max > im:
                 #               w_max = 10
                        elif hinge_position == "right":
                            h_min,h_max = xyxy[1], xyxy[3]
                            w_min,w_max = xyxy[0], xyxy[2]
                            h_min -= 20
                            if h_min < 0: # do not go beyond the image
                                h_min = 10
                           #h_max += 40
                            w_min -= 50  # do not go beyond the image
                            if w_min < 0:
                                w_min = 10
                            w_max += 20
                            xyxy[1], xyxy[3] = h_min,h_max
                            xyxy[0], xyxy[2] = w_min,w_max
                            fr_left_depth, fr_right_depth, frame_pl = frame_plane(im1ss[i], xyxy)

                        else:
                            h_min,h_max = xyxy[1], xyxy[3]
                            w_min,w_max = xyxy[0], xyxy[2]
                            h_min -= 20
                            if h_min < 0: # do not go beyond the image
                                h_min = 10
                            #h_max += 40
                            w_min -= 30  # do not go beyond the image
                            if w_min < 0:
                                w_min = 10
                            w_max += 30
                            xyxy[1], xyxy[3] = h_min,h_max
                            xyxy[0], xyxy[2] = w_min,w_max
                            fr_left_depth, fr_right_depth, frame_pl = frame_plane(im1ss[i], xyxy)

                            print("fr_left_depth::::::::::::"+str(fr_left_depth))
                            print("pl_left_depth::::::::::::"+str(pl_left_depth))
                            print("fr_right_depth::::::::::::"+str(fr_right_depth))
                            print("pl_right_depth::::::::::::"+str(pl_right_depth))
                            global push_pull_state
                            
                            if (abs(int(fr_left_depth) - int(pl_left_depth)) < 4) and (abs(int(fr_right_depth) - int(pl_right_depth)) < 4) :
                                push_pull_state = "closed"
                            elif (abs(int(fr_left_depth) - int(pl_left_depth)) > abs(int(fr_right_depth) - int(pl_right_depth))) and (int(fr_left_depth)<int(pl_left_depth)) :
                                push_pull_state = "left_push"
                            elif (abs(int(fr_left_depth) - int(pl_left_depth)) > abs(int(fr_right_depth) - int(pl_right_depth))) and (int(fr_left_depth)>int(pl_left_depth)) :
                                push_pull_state = "left_pull"
                            elif (abs(int(fr_left_depth) - int(pl_left_depth)) < abs(int(fr_right_depth) - int(pl_right_depth))) and (int(fr_right_depth)<int(pl_right_depth)) :
                                push_pull_state = "right_push"
                            elif (abs(int(fr_left_depth) - int(pl_left_depth)) < abs(int(fr_right_depth) - int(pl_right_depth))) and (int(fr_right_depth)>int(pl_right_depth)) :
                                push_pull_state = "right_pull"
                            else:
                                push_pull_state = "closed"
 
                            global push_sign
                            if push_pull_state == "left_push" or push_pull_state == "right_push":
                                push_sign = 1
                            elif push_pull_state == "left_pull" or push_pull_state == "right_pull":
                                push_sign = -1
                            else:
                                push_sign = 0

                            if push_pull_state == "left_push" or push_pull_state == "left_pull":
                                hinge_position  = "right"
                            elif push_pull_state == "right_push" or push_pull_state == "right_pull":
                                hinge_position = "left"
                            else:
                                hinge_position = "unknown"
 
                        print("hinge_position:::::::::::::::::"+str(hinge_position))
                        #print("push_pull_state:::::::::::::::::"+str(push_pull_state))
                        dot_product = np.dot(door_pl,frame_pl)
                        angle = push_sign * math.acos(dot_product)
                        global door_belief
                        if door_belief ==0:
                            door_belief = angle
                        else:
                            door_belief = 0.9 * door_belief+ 0.1* angle
                        
                        print("angle difference:::::::::::::::"+str(door_belief))
                        
                        #print("xyxy"+str(int(xyxy[0])))"""
                    # print("door_flap.HB_x:::::"+str(door_flap.HB_x))
                    # print("door_flap.HT_x:::::"+str(door_flap.HT_x))
                    # print("door_flap.OT_x:::.::"+str(door_flap.OT_x))
                    # print("door_flap.OB_x:::::"+str(door_flap.OB_x))

                    # print("door_flap.HB_y:::::"+str(door_flap.HB_y))
                    # print("door_flap.HT_y:::::"+str(door_flap.HT_y))
                    # print("door_flap.OT_y:::.::"+str(door_flap.OT_y))
                    # print("door_flap.OB_y:::::"+str(door_flap.OB_y))
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

       # Print time (inference + NMS)
        print(f'{s}Done. ({t2 - t1:.3f}s)')
        print("check1")
        if view_img:
            pass
            try:
                window_name = 'image'
                cv2.startWindowThread()
                cv2.imshow(window_name, im0)
                if (cv2.waitKey(30) >= 0): 
                    break
            except:
                cv2.destroyAllWindows()
        print("check2")
        # print("handle_position::"+str(handle_position))
        # print("push_pull_state::"+str(push_pull_state))

        door_pl, push_pull_state, data_integrity,orthogonal_flag = door_plane(im1ss[i], door_flap)
        if data_integrity == False:
            continue
        if orthogonal_flag == True:
            print("door_belief:::" + str(push_sign*90))
            print("Door is widely open")
            continue

        fr_left_depth, fr_right_depth, frame_pl, data_integrity = frame_plane(im1ss[i], door_frame)
        if data_integrity == False:
            continue

        dot_product = np.dot(door_pl,frame_pl)
        angle = push_sign * math.acos(dot_product)
        global door_belief
        if door_belief ==0:
            door_belief = angle
        else:
            door_belief = 0.9 * door_belief+ 0.1* angle

        print("=======================================================================")    
        print("Angle between door flap and door frame is :::"+str(push_sign * (90+door_belief/3.14*180)))
        print("+ sign represents that it is push door, - represents a pull door")
        print("door_pl :::"+str(door_pl))
        print("door_pl_center :::("+str(0.5*(door_flap.HT_x+ door_flap.OB_x))+", "+str(0.5*(door_flap.HT_y+ door_flap.OB_y))+", "+ str(im1ss[i][int(0.5*(door_flap.HT_y+ door_flap.OB_y)), int(0.5*(door_flap.HT_x+ door_flap.OB_x))])+")")
        print("frame_pl :::"+str(frame_pl))
        print("frame_pl_center :::("+str(0.5*(door_frame.HT_x+ door_frame.OB_x))+", "+str(0.5*(door_frame.HT_y+ door_frame.OB_y))+")")
        print("======================================================================")    

        print("check3")
       # Stream results
        if view_img:
            pass
            window_name = 'image'
            cv2.imshow(window_name, im0)
            if (cv2.waitKey(30) >= 0): 
                break
        print("check4")
            # Save results (image with detections)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    check_requirements()
    door_belief = 0
    handle_position = "unknown"
    push_pull_state = "unknown"
    door_plane_left = 0
    door_plane_right = 0
    push_sign = 0
    num_door = 0
    num_hinge = 0
    num_handle = 0
    hinge_array =[]
    handle_array =[]
    door_flap = door_body()
    door_frame = door_body()

    

    rospy.init_node('door_detection', anonymous=True)


    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            depth_sub = message_filters.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, queue_size = 1, buff_size=2**24)
            image_sub = message_filters.Subscriber("/camera/color/image_raw", Image, queue_size = 1, buff_size=2**24)
            rate = rospy.Rate(30)
            ats = message_filters.ApproximateTimeSynchronizer([image_sub, depth_sub], 10,1, reset =True)
            ats.registerCallback(detect)
            while not rospy.is_shutdown():
                rate.sleep()
                #rospy.spin()

