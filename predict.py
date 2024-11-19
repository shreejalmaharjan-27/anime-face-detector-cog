from cog import BasePredictor, Input, Path, BaseModel
import torch
import cv2
import shutil
import os
from pathlib import Path as PathLib
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords, plot_one_box
from utils.datasets import LoadImages
from utils.torch_utils import select_device
import random
from utils.google_utils import gdrive_download

class Output(BaseModel):
    file: Path
    txt: str

class Predictor(BasePredictor):
    def setup(self) -> None:
        self.device = select_device('')
        
        # download the model checkpoint if it does not exist
        if not os.path.exists("checkpoints/yolov5x_anime.pt"):
             gdrive_download('1-MO9RYPZxnBfpNiGY6GdsqCeQWYNxBdl','checkpoints/yolov5x_anime.pt')

        self.model = attempt_load("checkpoints/yolov5x_anime.pt", map_location=self.device)
        if self.device.type != 'cpu':
            self.model.half()
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]

    def predict(
        self,
        image: Path = Input(description="Input image"),
        size: int = Input(default=640, description="Size of the image"),
        confidence: float = Input(default=0.4, description="Confidence threshold"),
        iou: float = Input(default=0.5, description="IOU threshold"),
    ) -> Output:
        try:
            out_dir = 'inference/output'
            if os.path.exists(out_dir):
                shutil.rmtree(out_dir)
            os.makedirs(out_dir)

            imgsz = check_img_size(size, s=self.model.stride.max())
            dataset = LoadImages(str(image), img_size=imgsz)

            for path, img, im0s, _ in dataset:
                img = torch.from_numpy(img).to(self.device)
                img = img.half() if self.device.type != 'cpu' else img.float()
                img /= 255.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)

                with torch.no_grad():
                    pred = self.model(img, augment=False)[0]
                pred = non_max_suppression(pred, confidence, iou)
                det = pred[0]
                im0 = im0s.copy()
                txt_content = []

                if len(det):
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    for *xyxy, conf, cls in det:
                        # Convert tensors to float values
                        x1, y1, x2, y2 = [float(x.cpu().numpy()) for x in xyxy]
                        cls_id = int(cls)
                        conf_val = float(conf)
                        
                        label = f"{self.names[cls_id]} {conf_val:.2f}"
                        txt_content.append(f"{cls_id} {conf_val:.2f} {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f}")
                        plot_one_box(xyxy, im0, label=label, color=self.colors[cls_id])

                output_path = str(image).replace("/tmp", "", 1).replace("/", "")
                img_out = f"{out_dir}/{output_path}"
                txt_out = f"{out_dir}/{PathLib(output_path).stem}.txt"

                cv2.imwrite(img_out, im0)
                if txt_content:
                    with open(txt_out, 'w') as f:
                        f.write('\n'.join(txt_content))

                return Output(
                    file=Path(img_out),
                    txt='\n'.join(txt_content) if txt_content else ""
                )

        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()