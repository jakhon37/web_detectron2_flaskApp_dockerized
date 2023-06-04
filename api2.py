from flask import Flask
from flask import render_template
from flask import request
import os
from Segmentation.infer_pp import panoptic_on_image
from Segmentation.infer_pp import predict_tor
from detectron2.engine import DefaultPredictor


app = Flask(__name__)
UPLOAD_FOLDER = "/home/jakhon37/myProjects/microservices/web_dt/static"
@app.route("/", methods=["GET", "POST"])
def upload_predict():
    if request.method == "POST":
        image_file = request.files["image"]
        if image_file:
            image_location = os.path.join(
                UPLOAD_FOLDER,
                image_file.filename
            )
            image_file.save(image_location)
            print('image  ',image_file)
            # mask, img = panoptic_on_image(image_file, predictor, args.output)
            mask, img = predict_tor(image_location)

            return render_template("index.html", prediction  = 1 )
    return render_template("index.html", prediction  = 0 )

if __name__ == "__main__":
    
    app.run(port=22000, debug = True)