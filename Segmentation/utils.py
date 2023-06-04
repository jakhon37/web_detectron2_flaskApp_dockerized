
from detectron2.data import DatasetCatalog, MetadataCatalog
# from detectron2.utils.visualizer import Visualizer
from customVisualizer import Visualizer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import ColorMode
import random
import cv2
import PIL
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import datetime
import torch
import json
import os
def plot_samples(dataset_name, n=1):
    dataset_custom = DatasetCatalog.get(dataset_name)
    dataset_custom_metadata = MetadataCatalog.get(dataset_name)

    for s in random.sample(dataset_custom, n):
        img = cv2.imread(s["file_name"])
        v = Visualizer(img[:, :, ::-1],
                       metadata=dataset_custom_metadata,
                       scale=0.5)
        v = v.draw_dataset_dict(s)
        plt.figure(figsize=(15, 20))
        plt.imshow(v.get_image())
        plt.show()


def get_train_cfg(config_file_path, checkpoint_url, num_classes, device,train_dataset_name, val_dataset_name,
                  output_dir):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config_file_path))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(checkpoint_url)
    cfg.DATASETS.TRAIN = train_dataset_name
    # cfg.DATASETS.TEST = val_dataset_name
    cfg.DATASETS.VAL = val_dataset_name
    cfg.DATALOADER.NUM_WORKERS = 1
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.MAX_ITER = 2500
    # cfg.TEST.EVAL_PERIOD = 20
    cfg.SOLVER.STEPS = []
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.DEVICE = device
    cfg.OUTPUT_DIR = output_dir
    return cfg


from detectron2.engine import HookBase
from detectron2.data import build_detection_train_loader
import detectron2.utils.comm as comm




class ValidationLoss(HookBase):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.clone()
        self.cfg.DATASETS.TRAIN = cfg.DATASETS.VAL
        self._loader = iter(build_detection_train_loader(self.cfg))
        
    def after_step(self):
        data = next(self._loader)
        with torch.no_grad():
            loss_dict = self.trainer.model(data)
            
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {"val_" + k: v.item() for k, v in 
                                 comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                self.trainer.storage.put_scalars(total_val_loss=losses_reduced, 
                                                 **loss_dict_reduced)


def on_image(image_path, predictor, output_dir):
    
    now = datetime.datetime.now()        # creating new file name with date attached
    currentDate = "_" + str(now.day) + "_" + str(now.hour) + "_" + str(now.minute) + "_" + str(now.second)
    org, fontScale, thickness, font_face, color= (30, 40), 1, 1, cv2.FONT_HERSHEY_DUPLEX, (50, 50, 250)

    # if "jpg" "jpeg" "png" in image_path:
    # if image_path.is_dir():
    if os.path.isfile(image_path):
        print('hi')
        image = image_path 
        im = cv2.imread(image)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        print('Processing the input image... ' )
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
        for file in os.listdir(image_path):
            image = image_path + file
            im = cv2.imread(image)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            print('Processing the input image... ' )
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
    print('Successfully saved the Proceeded output image')
    print(f' output folder : {output_dir}')




 
def get_boxx_mask_img(image_path, predictor, output_dir):   #. get boxx croped with mask cropped
    im = cv2.imread(image_path)
    img_name = image_path.split('/')[-1][:-4]
    # print('imageeeee ', im)
    print(f'img shape{im.shape}')
    print('Processing the input image for mask... ' )
    outputs = predictor(im)
    resultmask = outputs["instances"].pred_masks.cpu().numpy()*255
    boxes = outputs["instances"].pred_boxes         # .detach().cpu().numpy()
    count, count_out = 0, 0 
    for ii, box in zip(range(len(boxes)), boxes):
        box = box.detach().cpu().numpy()
        x_top_left = box[0] #.cpu().numpy()
        y_top_left = box[1]
        x_bottom_right = box[2]
        y_bottom_right = box[3]
        width = (x_bottom_right-x_top_left)
        height = (y_bottom_right - y_top_left) 
        # print(f'height- {height}, width- {width}, \nx_top_left- {x_top_left}, y_top_left- {y_top_left}, \nx_bottom_right- {x_bottom_right}, y_bottom_right- {y_bottom_right}')
        print(f'height- {height}, width- {width}, \nx_top_left- {x_top_left}, y_top_left- {y_top_left}, \nx_bottom_right- {x_bottom_right}, y_bottom_right- {y_bottom_right}')
        detect_range_in_per = 50
        # if y_top_left > im.shape[0]*(1-(detect_range_in_per/100)) and width > 50 and height > 50:
        if y_top_left >  width > 50 and height > 50:
            count += 1
            # if width > im.shape[0]*(1-(detect_range_in_per/100)):
            crop_img_maskboxx = resultmask[ii][int(y_top_left): int(y_top_left + height), int(x_top_left ):int(x_top_left+ width)]
            crop_img_boxx = im[int(y_top_left): int(y_top_left + height), int(x_top_left ):int(x_top_left+ width)]
            # mask_n, box_n, mask_box_n = output_dir + (f'mask_n{ii}H{int(y_top_left)}.jpg'), output_dir + (f'box_n{ii}H{int(y_top_left)}.jpg'), output_dir + (f'mask_box_n{ii}H{int(y_top_left)}.png')
            box_n, mask_box_n =  output_dir + (f'{ii}{img_name}.png'), output_dir + (f'{ii}{img_name}_mask.png')
            cv2.imwrite(mask_box_n, crop_img_maskboxx)
            cv2.imwrite(box_n, crop_img_boxx)
            # cv2.imwrite(mask_n, resultmask[ii])
        else:
            count_out += 1

    print('Successfully saved the boxx img')
    # print(f'filtered: with {count} without{count_out}')
    return count, count_out


def get_boxx_mask_img_final(image_path, predictor, output_dir,):   ##### final full versionn so far: 3 types of img: just box, cropeed mask, merged boxmask
    im = cv2.imread(image_path)
    print(f'img shape{im.shape}')
    print('Processing the input image for mask... ' )
    outputs = predictor(im)
    resultmask = outputs["instances"].pred_masks.cpu().numpy()*255
    print('masks len : ', resultmask.shape)
    # print('masks: ', resultmask)
    """ 
    for single_mask in resultmask:
        print('single_mask. : ', single_mask)
        print('single_mask. len : ', len(single_mask))
         
        print('single_mask. len with index : ', len(single_mask))
        print('single_mask. sum : ', sum(single_mask))
        print('single_mask. double sum : ', sum(sum(single_mask)))
        print('single_mask. double sum2 : ', sum(sum(single_mask)[0:len((single_mask)/2)]))
        
        num_row = []

        for leneach, each_row in  zip(range(len(single_mask)), single_mask):
            if leneach > len(single_mask) / 15 and sum(each_row)>0:
                # print('sum of each_row in single_mask ', sum(each_row))
                num_row.append(each_row)
        print('len of num row: ', len(num_row))
            # else:
            #     print('no right conditions')
        """
                

    
    boxes = outputs["instances"].pred_boxes         # .detach().cpu().numpy()
    # print('box in final ', boxes)
    for ii, box in zip(range(len(boxes)), boxes):
        box = box.detach().cpu().numpy()
        x_top_left = box[0] #.cpu().numpy()
        y_top_left = box[1]
        x_bottom_right = box[2]
        y_bottom_right = box[3]
        width = (x_bottom_right-x_top_left)
        height = (y_bottom_right - y_top_left) 
        # print(f'height- {height}, width- {width}, \nx_top_left- {x_top_left}, y_top_left- {y_top_left}, \nx_bottom_right- {x_bottom_right}, y_bottom_right- {y_bottom_right}')
        print(f'height- {height}, width- {width}, \nx_top_left- {x_top_left}, y_top_left- {y_top_left}, \nx_bottom_right- {x_bottom_right}, y_bottom_right- {y_bottom_right}')
        detect_range_in_per = 50
        if y_top_left > im.shape[0]*(1-(detect_range_in_per/100)) and width > 50 and height > 50:
            # if width > im.shape[0]*(1-(detect_range_in_per/100)):
            crop_img_maskboxx = resultmask[ii][int(y_top_left): int(y_top_left + height), int(x_top_left ):int(x_top_left+ width)]
            crop_img_boxx = im[int(y_top_left): int(y_top_left + height), int(x_top_left ):int(x_top_left+ width)]
            # mask_n, box_n, mask_box_n = output_dir + (f'mask_n{ii}H{int(y_top_left)}.jpg'), output_dir + (f'box_n{ii}H{int(y_top_left)}.jpg'), output_dir + (f'mask_box_n{ii}H{int(y_top_left)}.png')
            mask_n, box_n, mask_box_n = output_dir + (f'{ii}mask_nH.jpg'), output_dir + (f'{ii}box_nH.jpg'), output_dir + (f'{ii}box_mask_nH.png')
            cv2.imwrite(mask_box_n, crop_img_maskboxx)
            cv2.imwrite(box_n, crop_img_boxx)
            # cv2.imwrite(mask_n, resultmask[ii])
    print('Successfully cropeed the mask and box and saved')

    boxlist = []
    masklist = []
    for file in os.listdir(output_dir):
        # print(f'name index {str(file)[ -2: -7]}')
        if 'box_n' in file:
            boxlist.append(file)
        elif 'box_mask_n' in file:
            masklist.append(file)
    boxlist.sort()
    masklist.sort()
    for bx, ms, lennn in zip(boxlist, masklist, range(len(boxlist))):
        imgbx = cv2.imread(output_dir + bx)
        imgms = cv2.imread(output_dir + ms)
        print(f'file nname{bx}: {imgbx.shape} and {ms}: {imgms.shape}')
        org_img_masked = cv2.bitwise_and(imgms, imgbx)
        contrastim = org_img_masked.std()
        print('contrast level of MaskBoxOrg_n', lennn , 'is: ', contrastim)
        mask_n = output_dir + (f'{bx[0:2]}MaskBoxOrg_n{lennn}.png')
        cv2.imwrite(mask_n, org_img_masked)
    print('Successfully merged the mask with box and saved the 3 types of img: just box, cropeed mask, merged boxmask')
    


    
def get_boxx_mask_img32(output_dir):     # read images from outfile and merge mask and boxx
    boxlist = []
    masklist = []
    for file in os.listdir(output_dir):
        # print(f'name index {str(file)[ -2: -7]}')
        if 'box_n' in file:
            boxlist.append(file)
        elif 'box_mask_n' in file:
            masklist.append(file)
    boxlist.sort()
    masklist.sort()
    for bx, ms, lennn in zip(boxlist, masklist, range(len(boxlist))):
        imgbx = cv2.imread(output_dir + bx)
        imgms = cv2.imread(output_dir + ms)
        print(f'file nname{bx}: {imgbx.shape} and {ms}: {imgms.shape}')
        org_img_masked = cv2.bitwise_and(imgms, imgbx)
        mask_n = output_dir + (f'MaskBoxOrg_n{lennn+2}.png')
        cv2.imwrite(mask_n, org_img_masked)
    print('Successfully saved the masked org boxx img')



def get_boxx_mask_img4(image_path, predictor, output_dir,): ###########. did not worked ## to do  if have time
    im = cv2.imread(image_path)
    print(f'img shape{im.shape}')
    print('Processing the input image for mask... ' )
    outputs = predictor(im)
    resultmask = outputs["instances"].pred_masks.cpu().numpy()*255
    boxes = outputs["instances"].pred_boxes         # .detach().cpu().numpy()
    for ii, box in zip(range(len(boxes)), boxes):
        box = box.detach().cpu().numpy()
        x_top_left = box[0] #.cpu().numpy()
        y_top_left = box[1]
        x_bottom_right = box[2]
        y_bottom_right = box[3]
        width = (x_bottom_right-x_top_left)
        height = (y_bottom_right - y_top_left) 
        # print(f'height- {height}, width- {width}, \nx_top_left- {x_top_left}, y_top_left- {y_top_left}, \nx_bottom_right- {x_bottom_right}, y_bottom_right- {y_bottom_right}')
        detect_range_in_per = 50
        if y_top_left > im.shape[0]*(1-(detect_range_in_per/100)) and width > 50 and height > 50:
            # if width > im.shape[0]*(1-(detect_range_in_per/100)):
            crop_img_maskboxx = resultmask[ii][int(y_top_left): int(y_top_left + height), int(x_top_left ):int(x_top_left+ width)]
            crop_img_boxx = im[int(y_top_left): int(y_top_left + height), int(x_top_left ):int(x_top_left+ width)]
            # mask_n, box_n, mask_box_n = output_dir + (f'mask_n{ii}H{int(y_top_left)}.jpg'), output_dir + (f'box_n{ii}H{int(y_top_left)}.jpg'), output_dir + (f'mask_box_n{ii}H{int(y_top_left)}.png')
            mask_n, box_n, mask_box_n = output_dir + (f'mask_n{ii}H.jpg'), output_dir + (f'box_n{ii}H.jpg'), output_dir + (f'box_mask_n{ii}H.png')
            cv2.imwrite(mask_box_n, crop_img_maskboxx)
            cv2.imwrite(box_n, crop_img_boxx)
            # cv2.imwrite(mask_n, resultmask[ii])

            
            # print(f'file nname{crop_img_boxx}: {crop_img_boxx.shape} and {crop_img_maskboxx}: {crop_img_maskboxx.shape}')
            print(f'file nname: {crop_img_boxx.shape} and : {crop_img_maskboxx.shape}')
            # cv2.imwrite(mask_n, resultmask[ii])
            img_float32 = np.float32(crop_img_maskboxx)
            rgb = cv2.cvtColor(img_float32, cv2.COLOR_GRAY2RGB)
            # imgms = cv2.imwrite(mask_box_n+'hi.jpg', rgb)

            imgbx = (crop_img_boxx)
            imgms = cv2.cvtColor(img_float32, cv2.COLOR_GRAY2RGB)

            print(f'file nname: {imgbx.shape} and : {imgms.shape}')
            # print(f'file nname{crop_img_boxx}: {crop_img_boxx.shape} and {crop_img_maskboxx}: {crop_img_maskboxx.shape}')
            org_img_masked = cv2.bitwise_and(imgms, imgbx)
            mask_n = output_dir + (f'MaskBoxOrg_new{{ii}}.png')
            cv2.imwrite(mask_n, org_img_masked)
    # boxlist = []
    # masklist = []

    print('Successfully saved the boxx img')
    
    
def get_boxx_mask_img_final_with_pp(image_path, predictor, output_dir,):   ##### final full versionn so far: 3 types of img: just box, cropeed mask, merged boxmask
    im = cv2.imread(image_path)
    print(f'img shape{im.shape}')
    print('Processing the input image for mask... ' )
    outputs = predictor(im)
    resultmask = outputs["instances"].pred_masks.cpu().numpy()*255
    print('masks len : ', resultmask.shape)

    boxes = outputs["instances"].pred_boxes         # .detach().cpu().numpy()
    # print('box in final ', boxes)
    for ii, box in zip(range(len(boxes)), boxes):
        box = box.detach().cpu().numpy()
        x_top_left = box[0] #.cpu().numpy()
        y_top_left = box[1]
        x_bottom_right = box[2]
        y_bottom_right = box[3]
        width = (x_bottom_right-x_top_left)
        height = (y_bottom_right - y_top_left) 
        # print(f'height- {height}, width- {width}, \nx_top_left- {x_top_left}, y_top_left- {y_top_left}, \nx_bottom_right- {x_bottom_right}, y_bottom_right- {y_bottom_right}')
        print(f'height- {height}, width- {width}, \nx_top_left- {x_top_left}, y_top_left- {y_top_left}, \nx_bottom_right- {x_bottom_right}, y_bottom_right- {y_bottom_right}')
        detect_range_in_per = 50
        if y_top_left > im.shape[0]*(1-(detect_range_in_per/100)) and width > 50 and height > 50:
            # if width > im.shape[0]*(1-(detect_range_in_per/100)):
            crop_img_maskboxx = resultmask[ii][int(y_top_left): int(y_top_left + height), int(x_top_left ):int(x_top_left+ width)]
            crop_img_boxx = im[int(y_top_left): int(y_top_left + height), int(x_top_left ):int(x_top_left+ width)]
            # mask_n, box_n, mask_box_n = output_dir + (f'mask_n{ii}H{int(y_top_left)}.jpg'), output_dir + (f'box_n{ii}H{int(y_top_left)}.jpg'), output_dir + (f'mask_box_n{ii}H{int(y_top_left)}.png')
            mask_n, box_n, mask_box_n, mask_rev_n = output_dir + (f'mask_n{ii}H.jpg'), output_dir + (f'box_n{ii}H.jpg'), output_dir + (f'box_mask_n{ii}H.png'), output_dir + (f'mask_rev{ii}H.png') 
            cv2.imwrite(mask_box_n, crop_img_maskboxx) # save mask
            cv2.imwrite(box_n, crop_img_boxx)
            # cv2.imwrite(mask_n, resultmask[ii])
    print('Successfully cropeed the mask and box and saved')


    masklist1 = []

    for file in os.listdir(output_dir):
        if 'box_mask_n' in file:
            masklist1.append(file)
            # elif 'box_mask_rev' in file:
    masklist1.sort()
    print('len mask ; ' , len(masklist1))
    for ms, lennn in zip(masklist1, range(len(masklist1))): # , masklist_rev
        imgmask = cv2.imread(output_dir + ms)
        crop_img_maskboxx_rev = cv2.bitwise_not(imgmask)
        mask_n_reversed = output_dir + (f'mask_reversed{lennn}H.png')
        cv2.imwrite(mask_n_reversed, crop_img_maskboxx_rev) # save reverse mask


    boxlist = []
    masklist = []
    masklist_rev = []
    for file in os.listdir(output_dir):
        # print(f'name index {str(file)[ -2: -7]}')
        if 'box_n' in file:
            boxlist.append(file)
        elif 'box_mask_n' in file:
            masklist.append(file)
        elif 'mask_reversed' in file:
            masklist_rev.append(file)
    
    boxlist.sort()
    masklist.sort()
    masklist_rev.sort()
    for bx, ms, ms_r, lennn in zip(boxlist, masklist, masklist_rev, range(len(boxlist))): # 
        imgbx = cv2.imread(output_dir + bx)
        imgms = cv2.imread(output_dir + ms)
        imgms_r = cv2.imread(output_dir + ms_r)
        print(f'file nname{bx}: {imgbx.shape} and {ms}: {imgms.shape} and {ms_r}: {imgms_r.shape}') # 
        org_img_masked = cv2.bitwise_and(imgms, imgbx)
        org_img_masked_rev = cv2.bitwise_and(imgms_r, imgbx)
        contrastim = org_img_masked.std()
        print('contrast level of MaskBoxOrg_n', lennn , 'is: ', contrastim)
        mask_n = output_dir + (f'MaskBoxOrg_n{lennn}.png')
        mask_n_rev = output_dir + (f'MaskBoxRev_n{lennn}.png')
        cv2.imwrite(mask_n, org_img_masked) # save masked img
        cv2.imwrite(mask_n_rev, org_img_masked_rev) # save reverse masked img 
    print('Successfully merged the mask with box and saved the 3 types of img: just box, cropeed mask, merged boxmask')
    



def on_video(video_path, predictor):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(
            'Error in opening the file...'
        )
        return
    (success, image) = cap.read()
    while success:
        predictions = predictor(image)
        v = Visualizer(image[:, :, ::-1], metadata={}, scale=0.5, instance_mode=ColorMode.SEGMENTATION)
        output = v.draw_instance_predictions(predictions['instances'].to('cpu'))

        cv2.imshow('Result', output.get_image()[:, :, ::-1])

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        (success, image) = cap.read()
