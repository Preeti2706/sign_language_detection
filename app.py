import sys, os, glob
from signLanguage.pipeline.training_pipeline import TrainPipeline
from signLanguage.exception import SignException
from signLanguage.utils.main_utils import decodeImage, encodeImageIntoBase64
from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS, cross_origin


app = Flask(__name__)
CORS(app)


class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"


def get_latest_model_path():
    """Fetch absolute path to latest best.pt model from artifacts folder"""
    artifacts_dir = os.path.abspath("artifacts")
    subfolders = [f.path for f in os.scandir(artifacts_dir) if f.is_dir()]
    
    if not subfolders:
        raise FileNotFoundError("No training artifacts found!")

    latest_folder = max(subfolders, key=os.path.getmtime)
    best_model_path = os.path.join(latest_folder, "model_trainer", "best.pt")
    
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"best.pt not found in {latest_folder}")
    
    return os.path.abspath(best_model_path)


def get_latest_prediction_image():
    """Find the latest prediction image path from yolov5/runs/detect"""
    detect_dir = os.path.join("yolov5", "runs", "detect")
    exp_folders = [f.path for f in os.scandir(detect_dir) if f.is_dir()]
    
    if not exp_folders:
        raise FileNotFoundError("No detection results found!")
    
    latest_exp = max(exp_folders, key=os.path.getmtime)
    img_path = os.path.join(latest_exp, "inputImage.jpg")
    
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Predicted image not found in {latest_exp}")
    
    return img_path


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/train")
def trainRoute():
    obj = TrainPipeline()
    obj.run_pipeline()
    return "Training Successfull!!"


@app.route("/predict", methods=['POST', 'GET'])
@cross_origin()
def predictRoute():
    try:
        image = request.json['image']
        decodeImage(image, clApp.filename)

        best_model_path = get_latest_model_path()

        os.system(f"cd yolov5 && python detect.py --weights \"{best_model_path}\" --img 416 --conf 0.5 --source ../data/inputImage.jpg")

        # get latest predicted file
        pred_img_path = get_latest_prediction_image()
        opencodedbase64 = encodeImageIntoBase64(pred_img_path)
        result = {"image": opencodedbase64.decode('utf-8')}

        # clean up runs folder safely (Windows + Linux)
        import shutil
        shutil.rmtree("yolov5/runs", ignore_errors=True)

    except Exception as e:
        print("Error:", e)
        return Response(str(e))

    return jsonify(result)



@app.route("/live", methods=['GET'])
@cross_origin()
def predictLive():
    try:
        best_model_path = get_latest_model_path()

        os.system(f"cd yolov5 && python detect.py --weights \"{best_model_path}\" --img 416 --conf 0.5 --source 0")
        os.system("rm -rf yolov5/runs")
        return "Camera starting!!"

    except Exception as e:
        print("Error:", e)
        return Response(str(e))


if __name__ == "__main__":
    clApp = ClientApp()
    app.run(host="0.0.0.0", port=8080)
