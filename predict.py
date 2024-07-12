# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path, BaseModel
import argparse
import subprocess

class Output(BaseModel):
    file: Path
    txt: str

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # self.model = torch.load("./weights.pth")

    def predict(
        self,
        image: Path = Input(description="Input image"),
        size: int = Input(default=640, description="Size of the image"),
        confidence: float = Input(default=0.4, description="Confidence threshold"),
        iou: float = Input(default=0.5, description="IOU threshold"),
    ) -> Output:
            parser = argparse.ArgumentParser()
            parser.add_argument('--weights', nargs='+', type=str, default='checkpoints/yolov5x_anime.pt', help='model.pt path(s)')
            parser.add_argument('--source', type=str, default=str(image), help='source')  # file/folder, 0 for webcam
            parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
            parser.add_argument('--img-size', type=int, default=size, help='inference size (pixels)')
            parser.add_argument('--conf-thres', type=float, default=confidence, help='object confidence threshold')
            parser.add_argument('--iou-thres', type=float, default=iou, help='IOU threshold for NMS')
            parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
            parser.add_argument('--view-img', action='store_true', help='display results')
            parser.add_argument('--save-txt', default=True, action='store_true', help='save results to *.txt')
            parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
            parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
            parser.add_argument('--augment', action='store_true', help='augmented inference')
            parser.add_argument('--update', action='store_true', help='update all models')
            opt = parser.parse_args([])
            # detect(opt=opt)
            inference_process = subprocess.Popen([
                 "python", 
                 "detect.py", 
                 "--weights", 
                 "checkpoints/yolov5x_anime.pt", 
                 "--source", str(image), 
                 "--output", "inference/output", 
                 "--img-size", str(size), 
                 "--conf-thres", str(confidence), 
                 "--iou-thres", str(iou), 
                 "--save-txt"
            ], stdout=subprocess.PIPE, universal_newlines=True)

            for line in iter(inference_process.stdout.readline, ''):
                print(line, end='')  # Print the output line by line
                if "Results saved" in line:
                    inference_process.kill()  # Kill the process if "Results saved" is found
                    break


            outputPath = str(image).replace("/tmp","", 1).replace("/", "")

            outputImage = Path(f"inference/output/{outputPath}")

            outputFile = outputPath.split(".")[0] + ".txt"

            try:
                with open(f"inference/output/{outputFile}", "r") as f:
                    txt = f.read()
            except FileNotFoundError:
                txt = ""

            return Output(file=outputImage,txt=txt)
       
