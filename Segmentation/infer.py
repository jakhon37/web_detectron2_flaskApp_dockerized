import argparse
import os
import numpy as np
import cv2
import pickle
import detectron2
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg, CfgNode
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2 import model_zoo
from detectron2.utils.visualizer import ColorMode
import random
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import datetime
import torch
import json

setup_logger()



def _get_parsed_args() -> argparse.Namespace:
    """
    Create an argument parser and parse arguments.
    :return: parsed arguments as a Namespace object
    """
    parser = argparse.ArgumentParser(description="Detectron2 demo")
    parser.add_argument(
        "--base_weight",       
        default= "./Segmentation/output/model_final.pth", 
        help="Base model weight to be used for inferance. "   
    )     
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device name (cpu/cudu)to be used for inferance. if not set - gpu"
    ) 
    parser.add_argument(
        "--task",
        default= "infer", # "mask",
        help="choose which task to process ->  infer / mask "
    )    
    parser.add_argument(
        "--class_num",
        default= (0 , 2), # "mask",
        help="choose which task to process ->  infer / mask "
    )  
    parser.add_argument(
        "--base_pkl",
        default= "./Segmentation/output/S_cfg_fance.pickle", 
        help="Base model pickle file to be used for detection. "            
    )
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--images",
    default="./Segmentation/input/guard_rail_data/val/", 
        nargs="+",         
        help="A list of space separated image files that will be processed. "
             "Results will be saved next to the original images with "
             "or a single glob pattern such as 'directory/*.jpg'"
             "'_processed_' appended to file name."
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--output",
        default="./output/infer_out/test/",
        help="A file or directory to save output visualizations. "
        "If not given, will save on default location.",
    )
    return parser.parse_args()

def setup_cfg(args):    
    args: argparse.Namespace = _get_parsed_args()
    cfg: CfgNode = pickle.load(open(args.base_pkl, 'rb'))  # args.base_pkl
    print('pickle file loaded')
    cfg.MODEL.WEIGHTS = os.path.join(args.base_weight)
    print('weights  loaded')
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.DEVICE = args.device
    print(f'device  set to {args.device}')
    cfg.freeze()
    return cfg

def on_image(image_path, predictor, output_dir):
    now = datetime.datetime.now()        # creating new file name with date attached
    currentDate = "_" + str(now.day) + "_" + str(now.hour) + "_" + str(now.minute) + "_" + str(now.second)
    org, fontScale, thickness, font_face, color= (30, 40), 1, 1, cv2.FONT_HERSHEY_DUPLEX, (50, 50, 250)
    if os.path.isfile(image_path):
        print('Processing the input image... ' )
        image = image_path 
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
        filename = output_dir + f'S{image_path[-8:-1]}_out.jpg'
        cv2.imwrite(filename, img)
    else:
        print('Processing the input images in folder... ' )
        for file in os.listdir(image_path):
            image = image_path + file
            #if filetype.is_image(filename): # import filetype
            print(f'image name: {image}')
            try:
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
                filename = output_dir + f'R{file}_out.jpg'
                cv2.imwrite(filename, img)
            except:
                print(f'{file} is not valid image')
    print('Successfully saved the Proceeded output image')
    print(f' output folder : {output_dir}')


def get_boxx_mask_img_final(image_path, predictor, output_dir, class_num):   ##### final full versionn so far: 3 types of img: just box, cropeed mask, merged boxmask
    im = cv2.imread(image_path)
    print(f'img shape{im.shape}')
    print('Processing the input image for mask... ' )
    outputs = predictor(im)
    resultmask = outputs["instances"].pred_masks.cpu().numpy()*255
    print('masks len : ', resultmask.shape)
    # print('masks: ', resultmask)
    
    boxes = outputs["instances"].pred_boxes         # .detach().cpu().numpy()
    _classes = outputs["instances"].pred_classes         # .detach().cpu().numpy()
    # print('class pred ', _classes)
    # print('box in final ', boxes)
    for ii, box, cls in zip(range(len(boxes)), boxes, _classes):
        box = box.detach().cpu().numpy()
        cls = cls.cpu().numpy()
        x_top_left = box[0] #.cpu().numpy()
        y_top_left = box[1]
        x_bottom_right = box[2]
        y_bottom_right = box[3]
        width = (x_bottom_right-x_top_left)
        height = (y_bottom_right - y_top_left) 
        # print(f'height- {height}, width- {width}, \nx_top_left- {x_top_left}, y_top_left- {y_top_left}, \nx_bottom_right- {x_bottom_right}, y_bottom_right- {y_bottom_right}')
        # print(f'height- {height}, width- {width}, \nx_top_left- {x_top_left}, y_top_left- {y_top_left}, \nx_bottom_right- {x_bottom_right}, y_bottom_right- {y_bottom_right}')
        detect_range_in_per = 99.999
        classes_to_proccess = class_num
        if y_top_left > im.shape[0]*(1-(detect_range_in_per/100)) and width > 1 and height > 1 and cls in classes_to_proccess:
            # if width > im.shape[0]*(1-(detect_range_in_per/100)):
            crop_img_maskboxx = resultmask[ii][int(y_top_left): int(y_top_left + height), int(x_top_left ):int(x_top_left+ width)]
            crop_img_boxx = im[int(y_top_left): int(y_top_left + height), int(x_top_left ):int(x_top_left+ width)]
            # mask_n, box_n, mask_box_n = output_dir + (f'mask_n{ii}H{int(y_top_left)}.jpg'), output_dir + (f'box_n{ii}H{int(y_top_left)}.jpg'), output_dir + (f'mask_box_n{ii}H{int(y_top_left)}.png')
            mask_n, box_n, mask_box_n = output_dir + (f'{image_path.split("/")[-1].split(".")[0]}.{ii}mask_nH.jpg'), output_dir + (f'{image_path.split("/")[-1].split(".")[0]}.{ii}box_nH.jpg'), output_dir + (f'{image_path.split("/")[-1].split(".")[0]}.{ii}msk_nH.png')
            cv2.imwrite(mask_box_n, crop_img_maskboxx)
            cv2.imwrite(box_n, crop_img_boxx)
            # cv2.imwrite(mask_n, resultmask[ii])
    print(f'Successfully cropeed the mask and box and saved in folder: {output_dir}')

    boxlist = []
    masklist = []
    for file in os.listdir(output_dir):
        # print(f'name index {str(file)[ -2: -7]}')
        if file.split(".")[0][0:-6] and 'box_nH.jpg' in file:
            boxlist.append(file)
        elif file.split(".")[0][0:-6] and'msk_nH.png' in file:
            masklist.append(file)
    boxlist.sort()
    masklist.sort()
    for bx, ms, lennn in zip(boxlist, masklist, range(len(boxlist))):
        imgbx = cv2.imread(output_dir + bx)
        imgms = cv2.imread(output_dir + ms)
        # print(f'file nname{bx}: {imgbx.shape} and {ms}: {imgms.shape}')
        org_img_masked = cv2.bitwise_and(imgms, imgbx)
        contrastim = org_img_masked.std()
        # print('contrast level of MaskBoxOrg_n', lennn , 'is: ', contrastim)
        mask_n = output_dir + (f'{bx.split(".")[0][0:-6]}MaskBoxOrg_n{lennn}.png')
        cv2.imwrite(mask_n, org_img_masked)
    print(f'Successfully merged the {len(masklist)}: masks with {len(boxlist)}: boxs and saved the 3 types of img: box crop, cropeed mask, merged boxmask / overal - {len(boxlist)*3}')
    



if __name__ == "__main__":
    args: argparse.Namespace = _get_parsed_args()
    cfg = setup_cfg(args)
    predictor = DefaultPredictor(cfg)
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    if args.task == "infer":
        on_image(args.images, predictor, args.output)
    if args.task == "mask":
        if os.path.isfile(args.images):
            get_boxx_mask_img_final(args.images, predictor, args.output, args.class_num)
        else:
            for file in os.listdir(args.images):
                image = args.images + file
                get_boxx_mask_img_final(image, predictor, args.output, args.class_num)
                
   
    
    # on_video(video_path, predictor)