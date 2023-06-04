from flask import Flask
from flask import render_template
from flask import request
import os
from Segmentation.infer_pp import panoptic_on_image
from Segmentation.infer_pp import predict_tor
from detectron2.engine import DefaultPredictor
import cv2

app = Flask(__name__, static_folder='static')
UPLOAD_FOLDER = "static" # raw_images
@app.route("/", methods=["GET", "POST"])
def upload_predict():
    if request.method == "POST":
        image_file = request.files["image"]
        if image_file:
            image_location = os.path.join(
                UPLOAD_FOLDER, 'raw_images',
                image_file.filename
            )
            image_file.save(image_location)
            print('image  ',image_file)
            # mask, img = panoptic_on_image(image_file, predictor, args.output)
            mask, img, f_name = predict_tor(image_location)
            filename_img = UPLOAD_FOLDER + f'/processed_images/{f_name.split("/")[-1].split(".")[0]}_out.jpg'
            img_name = f'{filename_img.split("/")[-1]}'
            print("image    :  ",filename_img)
            cv2.imwrite(filename_img, img)
            
            # print("f_name ",f_name)
            # image_location = os.path.join(UPLOAD_FOLDER, f_name)
            # img.save(image_location)
            raw_image_filenames = os.listdir(UPLOAD_FOLDER+'/raw_images')
            raw_image_filenames = sorted(raw_image_filenames, key=lambda x: os.path.getmtime(os.path.join(UPLOAD_FOLDER+'/raw_images', x)))
            prc_image_filenames = os.listdir(UPLOAD_FOLDER+'/processed_images')
            prc_image_filenames = sorted(prc_image_filenames, key=lambda x: os.path.getmtime(os.path.join(UPLOAD_FOLDER+'/processed_images', x)))

            return render_template("index.html", prediction  = 1, image_loc = image_file.filename, image_loc2 = img_name, r_i_f =  raw_image_filenames, p_i_f = prc_image_filenames)
    return render_template("index.html", prediction  = 0, image_loc = None, image_loc2 = None, r_i_f =  None, p_i_f = None  )

if __name__ == "__main__":
    
    
    app.run(host='0.0.0.0', port=22111, debug = True) # 