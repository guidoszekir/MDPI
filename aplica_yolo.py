import rclpy
from rclpy.node import Node
import os
from std_msgs.msg import String

from ros2_mediapipe_msgs.msg import MediapipePose
from ros2_mediapipe_msgs.msg import MediapipeHands

from geometry_msgs.msg import Twist

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2
from cv_bridge import CvBridge
from scipy.spatial import distance
from std_msgs.msg import Int16

from random import random

from tello_msgs.srv import TelloAction 

import time

import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from PIL import Image as IMG
import numpy as np

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

import torchvision.transforms as transforms

from ros2_yolo_msgs.msg import YoLoMSG
from ros2_yolo_msgs.msg import YoLoVetor



class agro_drone(Node):

    def __init__(self):
        super().__init__('agro_drone')
        self.bridge = []
        self.bridge = CvBridge()
        self.imagem = []
        
        self.troca_topico = 2 # 1 para UGV e 2 para UAV
        
        self.print_image_topic = 0;
        
        self.publisher_image_ = self.create_publisher(Image, 'YoLov7/image', 10)
        self.publisher_yolo_ = self.create_publisher(YoLoVetor, 'YoLoVetor/result', 10)
        #sub
#        self.subscription = self.create_subscription(
#            Image,
#            '/UGV_image',
#            #'/UAV/image',
#            self.image_callback,
#            0)
#        self.subscription  # prevent unused variable warning
        
        self.subscription2 = self.create_subscription(
            Image,
            #'/husky_image',
            '/UAV_image',
            self.image_callback2,
            0)
        self.subscription2  # prevent unused variable warning
        
#        self.subTroca = self.create_subscription(
#            Int16,
#            #'/husky_image',
#            '/YoLo_troca_imagem',
#            self.troca,
#            0)
#        self.subTroca  # prevent unused variable warning
        rclpy.spin_once(self)
        time.sleep(3)
        
        rclpy.spin_once(self)
        
        #with torch.no_grad():
        #    detect()
        with torch.no_grad():
                        self.detect()
                        
        
    def troca(self, msg):
        self.troca_topico = msg.data
#        print("--------------------------------------------------\n")
#        print(msg)
#        print("Valor de troca: "+str(self.troca_topico) )
#        print("--------------------------------------------------\n")
    	 
    def image_callback(self, msg):
        if self.troca_topico == 1:
                self.imagem = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')   
                self.imagem = cv2.flip(self.imagem, 1)
                self.imagem = cv2.resize(self.imagem, [640,480])


    def image_callback2(self, msg):
        if self.troca_topico == 2:
                self.imagem = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')   
                self.imagem = cv2.flip(self.imagem, 1)
                self.imagem = cv2.resize(self.imagem, [640,480])


    def detect(self,save_img=False):
            YoLoVetor_ = YoLoVetor()
    
            source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
#            save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
#            webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
#                ('rtsp://', 'rtmp://', 'http://', 'https://'))

            # Directories
#            save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
#            (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

            # Initialize
            set_logging()
            device = select_device(opt.device)
            half = device.type != 'cpu'  # half precision only supported on CUDA

            # Load model
            model = attempt_load(weights, map_location=device)  # load FP32 model
            stride = int(model.stride.max())  # model stride
            imgsz = check_img_size(imgsz, s=stride)  # check img_size

            if trace:
                model = TracedModel(model, device, opt.img_size)

            if half:
                model.half()  # to FP16

            # Second-stage classifier
            classify = False
            if classify:
                modelc = load_classifier(name='resnet101', n=2)  # initialize
                modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

            # Set Dataloader
            vid_path, vid_writer = None, None
            
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
#            dataset = LoadStreams(source, img_size=imgsz, stride=stride)
            
            # Get names and colors
            names = model.module.names if hasattr(model, 'module') else model.names
            colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

            # Run inference
            if device.type != 'cpu':
                model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
            old_img_w = old_img_h = imgsz
            old_img_b = 1

            t0 = time.time()
            while True:
#            for path, img, im0s, vid_cap in dataset:
                img = [] 
                transform = transforms.Compose([transforms.PILToTensor()])
#                print("Imagem da web "+str(type(img)))
#                print("Imagem do ROS "+str(type(IMG.fromarray( self.imagem))))
#                print("Imagem do ROS "+str(type(np.copy( self.imagem))))
#                img = np.copy( self.imagem)
                img = cv2.cvtColor(self.imagem, cv2.COLOR_BGR2RGB)

#                img = torch.from_numpy(img).to(device)
                img = transform(( IMG.fromarray( img))).to(device)
                img = img.half() if half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)
                    
                    

                # Warmup
                if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                    old_img_b = img.shape[0]
                    old_img_h = img.shape[2]
                    old_img_w = img.shape[3]
                    for i in range(3):
                        model(img, augment=opt.augment)[0]

                # Inference
                t1 = time_synchronized()
                pred = model(img, augment=opt.augment)[0]
                t2 = time_synchronized()

                # Apply NMS
                pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
                t3 = time_synchronized()

                # Apply Classifier
                if classify:
                    pred = apply_classifier(pred, modelc, img, im0s)

                # Process detections
                for i, det in enumerate(pred):  # detections per image
##                    if webcam:  # batch_size >= 1
##                        p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
#                        im0 = self.imagem
#                    else:
#                        p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

                    im0 = self.imagem
                    s = ""   
#                    p = Path(p)  # to Path
#                    save_path = str(save_dir / p.name)  # img.jpg
#                    txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                        # Print results
                        for c in det[:, -1].unique():
                            n = (det[:, -1] == c).sum()  # detections per class
                            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                            

                        print(det[0][0])
                        # Write results
                        for *xyxy, conf, cls in reversed(det):
                            YoLoMSG_ = YoLoMSG() 
                            YoLoMSG_.classe = f'{names[int(cls)]}'
                            YoLoMSG_.confianca = float(f'{conf:.2f}')
                            YoLoMSG_.x1 = int(xyxy[0])
                            YoLoMSG_.y1 = int(xyxy[1])
                            YoLoMSG_.x2 = int(xyxy[2])
                            YoLoMSG_.y2 = int(xyxy[3])
                            YoLoVetor_.detectado.append(YoLoMSG_)
                            if save_img or view_img:  # Add bbox to image
                                label = f'trap {conf:.2f}'
#                                print(label)
                                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
#                                print(label)
#                                print("xyxy: "+str(int(xyxy[0])))
#                                print("Label: "+str(label))

                        

                    # Print time (inference + NMS)
#                    print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

                    
                    # Stream results
                    if view_img:
                        rclpy.spin_once(self)
                        self.publisher_image_.publish(self.bridge.cv2_to_imgmsg(im0, "bgr8"))
#                        self.imagem = im0
#                        cv2.imshow("result", im0)
#                        cv2.waitKey(1)  # 1 millisecond


                    self.publisher_yolo_.publish(YoLoVetor_)
                    YoLoVetor_ = YoLoVetor()
#            print(f'Done. ({time.time() - t0:.3f}s)')

    





           

   

agro_drone_controll = []
        


imagem_ros = []

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='1', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)
    
    #detect()

    #main()
    rclpy.init()
    agro_drone_controll = agro_drone()
    rclpy.spin(agro_drone_controll)
            
     
    
  
