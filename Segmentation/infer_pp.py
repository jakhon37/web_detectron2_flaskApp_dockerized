import argparse
import datetime
import json
import os
import numpy as np
import cv2
import pickle
import detectron2
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg, CfgNode
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog, DatasetCatalog

setup_logger()
# from utils import *

def setup_cfg(args):
    # load config from file and command-line arguments
    # cfg = get_cfg()
    # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
    # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
    # add_panoptic_deeplab_config(cfg)
    # cfg.merge_from_file(args.config_file)
    # cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    # cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    # cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    # cfg.freeze()
    
    args: argparse.Namespace = _get_parsed_args()
    cfg = get_cfg()
    # cfg: CfgNode = pickle.load(open(args.base_pkl, 'rb'))  # args.base_pkl
    cfg.merge_from_file(model_zoo.get_config_file(args.config_file))# "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")) 
#           "Cityscapes/mask_rcnn_R_50_FPN": "142423278/model_final_af9cf5.pkl",
    # print('pickle file loaded')
    cfg.MODEL.WEIGHTS = os.path.join(args.base_weight)
    # print('weights  loaded')
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.DEVICE = args.device
    # print(f'device  set to {args.device}')
    cfg.freeze()
    return cfg


def _get_parsed_args() -> argparse.Namespace:
    """
    Create an argument parser and parse arguments.
    :return: parsed arguments as a Namespace object
    """
    parser = argparse.ArgumentParser(description="Detectron2 demo")
    parser.add_argument(
        "--base_weight", # https://dl.fbaipublicfiles.com/detectron2/Cityscapes/mask_rcnn_R_50_FPN/142423278/model_final_af9cf5.pkl
        default= "./Segmentation/weights/model_final_cafdb1.pkl" , # "/home/jakhon37/PROJECTS/dttn2/output/model_final_city.pkl", #voc.pkl", #city.pkl",    # /home/jakhon37/PROJECTS/dttn2/output/segmentation/11model_final.pth
        help="Base model weight to be used for inferance. "   # PascalVOC-Detection/faster_rcnn_R_50_C4.yaml
    )     
    parser.add_argument(
        "--device",
        default="cuda", # cuda
        help="Device name (cpu/cudu)to be used for inferance. if not set - gpu"
    )    
    parser.add_argument(
        "--config_file", # https://github.com/facebookresearch/detectron2/blob/main/detectron2/model_zoo/model_zoo.py
        default= "COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml", # "Cityscapes/mask_rcnn_R_50_FPN.yaml",  #faster_rcnn_R_50_C4.yaml",  #Cityscapes/mask_rcnn_R_50_FPN.yaml",             
        help="Base model pickle file to be used for detection. "        # PascalVOC-Detection/faster_rcnn_R_50_FPN.yaml
    )
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--images",
        # default="./dataset/testData/",    # /home/jakhon37/PROJECTS/dttn2/dataset/testData /home/jakhon37/PROJECTS/dttn2/dataset/raod_f/road_fc3.jpeg

      # default="./dataset/testData/s/201213_E_15_CCW_in_E_B_001_01937.png",    
        default= '/home/jakhon37/myProjects/microservices/web_dt/Segmentation/input/testData2/s/201213_E_15_CCW_in_E_B_001_01692.png',
        # "/home/jakhon37/myProjects/microservices/web_dt/Segmentation/input/testData2/s", 
        
        #/home/jakhon37/PROJECTS/dttn2/dataset/pp_seg/" ,#golf", # "/home/jakhon37/PROJECTS/dttn2/dataset/road_marks_seg/test/201222_E_11_CW_in_D_C_004_00276.png", # "/home/jakhon37/PROJECTS/dttn2/dataset/road_marks_seg/test/", #./dataset/raod_f/", #guardrail_test_data/", #./dataset/raod_f/",    # /home/jakhon37/PROJECTS/dttn2/dataset/testData2/s/201222_E_14_CCW_in_D_C_001_00660.png
       # /home/jakhon37/PROJECTS/dttn2/dataset/testData2/s/201213_E_15_CCW_in_N_B_000_00234.png
        # /home/jakhon37/PROJECTS/dttn2/dataset/testtt.jpg.  /home/jakhon37/PROJECTS/dttn2/dataset/testData/s
        nargs="+",          #. /home/jakhon37/PROJECTS/dttn2/dataset/testData/s/201213_E_15_CCW_in_E_B_001_01937.png.  /home/jakhon37/PROJECTS/dttn2/dataset/testData/s/201213_E_14_CCW_in_E_B_002_02589.png. 
        help="A list of space separated image files that will be processed. "
             "Results will be saved next to the original images with "
             "or a single glob pattern such as 'directory/*.jpg'"
             "'_processed_' appended to file name."
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--output",
        default="./Segmentation/output/panoptic_s/",
        help="A file or directory to save output visualizations. "
        "If not given, will save on default location.",
    )
    return parser.parse_args()

def panoptic_on_image(image_path, predictor, output_dir, cfg):
    img = np.zeros(5)
    if os.path.isfile(image_path):
        # print('Processing the input image... ' )
        image_n = image_path 
        im = cv2.imread(image_n)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        predictions, segmentInfo = predictor(im)["panoptic_seg"]

 
        
        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1) 
        out = v.draw_panoptic_seg_predictions(predictions.to("cpu"), segmentInfo, area_threshold=.1)
        img = (out.get_image()[:, :, ::-1])

        filename_img = output_dir + f'{image_path.split("/")[-1].split(".")[0]}_out.jpg'
        cv2.imwrite(filename_img, img)
        
        json_data = {}
        json_data["segmentInfo"] = segmentInfo
        json_name = output_dir + f'{image_path.split("/")[-1].split(".")[0]}_out.json'
        panoptic_info = (output_dir)
        with open (json_name, "w") as outfile:
            json.dump(json_data, outfile, indent="\t")
            
        prediction_numpy = predictions.cpu().numpy()
        filename = output_dir + f'{image_n.split("/")[-1].split(".")[0]}_panoptic_mask.jpg'
        cv2.imwrite(filename, prediction_numpy)
        print("image   name :  ",filename)

        return prediction_numpy, img, filename_img

    else:
        # try:
            prediction_numpy = []
            # print('Processing the input images in folder... ' )
            for image_n in os.listdir(image_path):
                # print(image_n)
                image = f'{image_path}/{image_n}'
                im = cv2.imread(image)
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                predictions, segmentInfo = predictor(im)["panoptic_seg"]
                v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1) 
                out = v.draw_panoptic_seg_predictions(predictions.to("cpu"), segmentInfo, area_threshold=.1)
                img = (out.get_image()[:, :, ::-1])
                filename = output_dir + f'{image_n.split(".")[0]}_out.jpg'
                cv2.imwrite(filename, img)

                # prediction_numpy = predictions.cpu().numpy()
                prediction_numpy.append(predictions.cpu().numpy())
                filename = output_dir + f'{image_n.split(".")[0]}_panoptic_mask.jpg'
                cv2.imwrite(filename, np.array(prediction_numpy))
                print("image   name :  ",filename)

            return prediction_numpy, img
        # except:                 
        #     print(f'{image_n} is not valid image')


    # print('Successfully saved the Proceeded output image')
    # print(f' output folder : {output_dir} with image: {image_n.split("/")[-1]}')
    return prediction_numpy, img, filename_img
    
def on_image(image_path, predictor, output_dir):
    
    now = datetime.datetime.now()        # creating new file name with date attached
    currentDate = "_" + str(now.day) + "_" + str(now.hour) + "_" + str(now.minute) + "_" + str(now.second)
    org, fontScale, thickness, font_face, color= (30, 40), 1, 1, cv2.FONT_HERSHEY_DUPLEX, (50, 50, 250)

    # if "jpg" "jpeg" "png" in image_path:
    # if image_path.is_dir():
    if os.path.isfile(image_path):
        # print('Processing the input image... ' )
        image_n = image_path 
        im = cv2.imread(image_n)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        outputs = predictor(im)
        # print(len(outputs))
        v = Visualizer(im[:, :, ::-1], metadata={}, scale=0.5, instance_mode=ColorMode.SEGMENTATION)
        image = v.draw_instance_predictions(outputs['instances'].to('cpu'))
        if len(outputs['instances']) >= 1:
            text = (" detected: " + str(len(outputs['instances'])))
        else:
            text = "no detection " 
        img = image.get_image()
        img = cv2.putText(img, text, org, font_face, fontScale, color, thickness, cv2.LINE_AA)
        filename = output_dir + f'S{image_path[-8:-1]}_out.jpg'
        cv2.imwrite(filename, img)
    else:
        print('Processing the input images in folder... ' )
        for image_n in os.listdir(image_path):
            # print(image_n)
            # print(image_n.shape())
            image = image_path + image_n
            im = cv2.imread(image)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            outputs = predictor(im)
            v = Visualizer(im[:, :, ::-1], metadata={}, scale=0.5, instance_mode=ColorMode.SEGMENTATION)
            image = v.draw_instance_predictions(outputs['instances'].to('cpu'))
            if len(outputs['instances']) >= 1:
                text = (" detected: " + str(len(outputs['instances'])))
            else:
                text = "no detection " 
            img = image.get_image()
            img = cv2.putText(img, text, org, font_face, fontScale, color, thickness, cv2.LINE_AA)
            filename = output_dir + f'R{image_n}_out.jpg'
            cv2.imwrite(filename, img)
    # print('Successfully saved the Proceeded output image')
    # print(f' output folder : {output_dir} with image: {image_n.split("/")[-1]}')

# import glob
# if __name__ == "__main__":
    # args: argparse.Namespace = _get_parsed_args()
    
    
    # folder = "/home/yolov5Analyze/MyData/res"
    # sub_folders = [name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))]

    # # for each folder
    # for i in range(13, len(sub_folders)):
        
    #     fname = sub_folders[i]
    #     print(f'==========================================================================\n{i} , {fname}')
        
    #     pth = f"{folder}/{fname}" 
         
    #     lst = glob.glob(f'{pth}/*.jpg') #os.listdir(pth) # your directory path
        
    #     nums = len(lst)
        
    #     os.makedirs(f"/home/PanopticSegmentation/res/{fname}", exist_ok=True)
    #     # for each image
    #     for k in range(nums):
    #         imgFile = f"{pth}/{k}.jpg"
            
    #         args.images = imgFile # "/home/yolov5Analyze/MyData/res/1 (1)/0.jpg"
                
    #         cfg = setup_cfg(args)
    #         predictor = DefaultPredictor(cfg)

    #         if not os.path.exists(args.output):
    #             os.makedirs(args.output, exist_ok=True)

    #         mask, img = panoptic_on_image(args.images, predictor, args.output)
    #         np.save(f"/home/Sherzod/PanopticSegmentation/res/{fname}/{k}", mask)
    #         print(img.shape)
    #         cv2.imwrite(f"/home/PanopticSegmentation/res/{fname}/{k}_r.jpg", img)
            
        # try:
        #     print(len(mask))
        #     for i in range(len(mask)):
        #         print(mask[i].shape)
        # except:
        #     print(mask.shape)
 
def predict_tor(image):
    print('start detectron ')
    args: argparse.Namespace = _get_parsed_args()
    cfg = setup_cfg(args)
    predictor = DefaultPredictor(cfg)
    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)
    mask, img, f_name = panoptic_on_image(image, predictor, args.output, cfg)
    
    # filename = args.output + f'R{image.split("/")[-1].split(".")[0]}_out.jpg'
    # print("image   name :  ",filename)
    # cv2.imwrite(filename, img)
            
    print('------------------saved--------------')
    
    print(f' output folder : {args.output} with image: {image.split("/")[-1]}')

    return mask, img, f_name

def predict_tor2():
    print('start detectron ')
    args: argparse.Namespace = _get_parsed_args()
    cfg = setup_cfg(args)
    predictor = DefaultPredictor(cfg)
    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)
    mask, img, f_name = panoptic_on_image(args.images, predictor, args.output, cfg)
    print('------------------saved--------------')
    return mask, img, f_name
    
if __name__ == "__main__":
    
    #predict_tor2()
    
    
    
    print('start detectron ')
    args: argparse.Namespace = _get_parsed_args()
    
    cfg = setup_cfg(args)
    predictor = DefaultPredictor(cfg)

    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)

    mask, img = panoptic_on_image(args.images, predictor, args.output)
    print('before save  ================')
    np.save(f"{args.output}/skyMask.npy", mask)
    print('------------------saved--------------')
