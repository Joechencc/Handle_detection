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

    
def detect(image, depth):
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
        print("hehehehehes")

        # Process detections
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
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            # Write results
            for *xyxy, conf, cls in reversed(det):
                if save_txt:  # Write to file
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                    with open(txt_path + '.txt', 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

                if save_img or view_img:  # Add bbox to image
                    label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

        # Print time (inference + NMS)
        print(f'{s}Done. ({t2 - t1:.3f}s)')

        # Stream results
        if view_img:
            window_name = 'image'
            cv2.imshow(window_name, im0)
            cv2.waitKey(1) # 1 millisecond

        # Save results (image with detections)
        if save_img:
            if dataset.mode == 'image':
                cv2.imwrite(save_path, im0)
            else:  # 'video'
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer

                    fourcc = 'mp4v'  # output video codec
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                vid_writer.write(im0)

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
            ats = message_filters.ApproximateTimeSynchronizer([image_sub, depth_sub], 10,1 )
            ats.registerCallback(detect)
            while not rospy.is_shutdown():
                rate.sleep()
                #rospy.spin()


