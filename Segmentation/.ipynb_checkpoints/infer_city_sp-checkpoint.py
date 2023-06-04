import argparse
import os
import numpy as np
import cv2
import pickle
import detectron2
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg, CfgNode

setup_logger()
from utils import *

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
    print('pickle file loaded')
    cfg.MODEL.WEIGHTS = os.path.join(args.base_weight)
    print('weights  loaded')
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.DEVICE = args.device
    print(f'device  set to {args.device}')
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
        default="/home/jakhon37/PROJECTS/dttn2/output/model_final_city.pkl", #voc.pkl", #city.pkl",    # /home/jakhon37/PROJECTS/dttn2/output/segmentation/11model_final.pth
        help="Base model weight to be used for inferance. "   # PascalVOC-Detection/faster_rcnn_R_50_C4.yaml
    )     
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device name (cpu/cudu)to be used for inferance. if not set - gpu"
    )    
    parser.add_argument(
        "--config_file", # https://github.com/facebookresearch/detectron2/blob/main/detectron2/model_zoo/model_zoo.py
        default="Cityscapes/mask_rcnn_R_50_FPN.yaml",  #faster_rcnn_R_50_C4.yaml",  #Cityscapes/mask_rcnn_R_50_FPN.yaml",             
        help="Base model pickle file to be used for detection. "        # PascalVOC-Detection/faster_rcnn_R_50_FPN.yaml
    )
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--images",
        # default="./dataset/testData/",    # /home/jakhon37/PROJECTS/dttn2/dataset/testData /home/jakhon37/PROJECTS/dttn2/dataset/raod_f/road_fc3.jpeg

    # default="./dataset/testData/s/201213_E_15_CCW_in_E_B_001_01937.png",    
    default="./dataset/raod_f/",    # /home/jakhon37/PROJECTS/dttn2/dataset/testData2/s/201222_E_14_CCW_in_D_C_001_00660.png
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
        default="./output/new/",
        help="A file or directory to save output visualizations. "
        "If not given, will save on default location.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args: argparse.Namespace = _get_parsed_args()
    cfg = setup_cfg(args)
    predictor = DefaultPredictor(cfg)
    """
    if "jpg" "png" in args.images:
        images = args.images
    else:
        for file in os.os.listdir(args.images):
            images = file
    """
    if not os.path.exists(args.output):
        os.makedirs("./output/new/", exist_ok=True)
    # get_boxx_mask_img_final(args.images, predictor, args.output)
    # get_boxx_mask_img_final_with_pp(args.images, predictor, args.output)
    on_image(args.images, predictor, args.output)
    # get_boxx_mask_img4(args.images, predictor, args.output)
    # get_boxx_mask_img32(args.output)
    """
    sumc = 0
    sumcout = 0
    for file in os.listdir(args.images):
        # print(file)
        if 'png' in file:
            image = (args.images + file)
            # print(image)
            # print(image.shape)
            c1, c2 = get_boxx_mask_img(image, predictor, args.output)
            sumc += c1
            sumcout += c2
    print(f'filtered: with {sumc} without{sumcout}')

    """
    # get_boxx_mask_img()
    
    # video_path = 'dataset/e_motor/test/'
    # on_video(video_path, predictor)