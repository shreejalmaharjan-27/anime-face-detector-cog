from cog import BasePredictor, Input, Path, BaseModel
import torch
import cv2
import os
from pathlib import Path as PathLib
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords, plot_one_box
from utils.datasets import LoadImages
from utils.torch_utils import select_device
from utils.google_utils import gdrive_download
import random
from functools import wraps
import gc

class Output(BaseModel):
    file: Path
    txt: str

class ModelHolder:
    model = None
    device = None

def print_gpu_memory():
    if torch.cuda.is_available():
        print(f"GPU Memory allocated: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
        print(f"GPU Memory cached: {torch.cuda.memory_reserved()/1024**2:.2f}MB")

def do_load(weights_path):
    global model_holder
    print("Loading model")
    print_gpu_memory()
    
    if not os.path.exists(weights_path):
        raise RuntimeError(f"Model weights not found at {weights_path}")

    model_holder = ModelHolder()
    model_holder.device = select_device('')
    
    try:
        model_holder.model = attempt_load(weights_path, map_location=model_holder.device)
        model_holder.is_half = False
        if model_holder.device.type != 'cpu':
            model_holder.model = model_holder.model.half()
            model_holder.is_half = True
        print("Model loaded successfully")
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")
    print_gpu_memory()

def make_mem_efficient(cls: BasePredictor):
    if not torch.cuda.is_available():
        return cls

    old_setup = cls.setup
    old_predict = cls.predict

    @wraps(old_setup)
    def new_setup(self, *args, **kwargs):
        ret = old_setup(self, *args, **kwargs)
        self._move_to("cpu")
        return ret

    @wraps(old_predict)
    def new_predict(self, *args, **kwargs):
        self._move_to("cuda")
        try:
            ret = old_predict(self, *args, **kwargs)
        finally:
            self._cleanup()
            self._move_to("cpu")
        return ret

    cls.setup = new_setup
    cls.predict = new_predict
    return cls

@make_mem_efficient
class Predictor(BasePredictor):
    def _move_to(self, device):
        if model_holder.model is not None:
            model_holder.model.to(device)
            if device == 'cuda' and not model_holder.is_half:
                model_holder.model = model_holder.model.half()
                model_holder.is_half = True
        torch.cuda.empty_cache()
        gc.collect()

    def _cleanup(self):
        torch.cuda.empty_cache()
        gc.collect()
        if hasattr(self, 'current_img'):
            del self.current_img
        if hasattr(self, 'current_pred'):
            del self.current_pred

    def setup(self) -> None:
        weights_path = "checkpoints/yolov5x_anime.pt"
        if not os.path.exists(weights_path):
            os.makedirs("checkpoints", exist_ok=True)
            try:
                gdrive_download('1-MO9RYPZxnBfpNiGY6GdsqCeQWYNxBdl', weights_path)
            except Exception as e:
                raise RuntimeError(f"Failed to download model weights: {str(e)}")
            
        do_load(weights_path)
        self.model = model_holder.model
        self.device = model_holder.device
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]

    def predict(
        self,
        image: Path = Input(description="Input image"),
        size: int = Input(default=640, description="Size of the image"),
        confidence: float = Input(default=0.4, description="Confidence threshold"),
        iou: float = Input(default=0.5, description="IOU threshold"),
    ) -> Output:
        out_dir = 'inference/output'
        os.makedirs(out_dir, exist_ok=True)
        print("Starting prediction")
        print_gpu_memory()

        # Validate input image
        if not os.path.exists(image):
            raise ValueError(f"Input image not found: {image}")

        try:
            imgsz = check_img_size(size, s=self.model.stride.max())
            dataset = LoadImages(str(image), img_size=imgsz)

            for path, img, im0s, _ in dataset:
                if img is None or im0s is None:
                    raise ValueError("Failed to load image")

                self.current_img = torch.from_numpy(img).to(self.device)
                if self.device.type != 'cpu':
                    self.current_img = self.current_img.half()
                self.current_img /= 255.0
                
                if self.current_img.ndimension() == 3:
                    self.current_img = self.current_img.unsqueeze(0)

                print("Before inference")
                print_gpu_memory()

                with torch.no_grad():
                    self.current_pred = self.model(self.current_img, augment=False)[0]
                    if self.current_pred is None:
                        raise RuntimeError("Model inference failed")
                        
                    self.current_pred = non_max_suppression(self.current_pred, confidence, iou)
                    if self.current_pred is None:
                        raise RuntimeError("Non-max suppression failed")

                print("After inference")
                print_gpu_memory()

                det = self.current_pred[0]
                if det is None:
                    # No detections found, return empty result
                    output_path = str(image).replace("/tmp", "", 1).replace("/", "")
                    img_out = f"{out_dir}/{output_path}"
                    cv2.imwrite(img_out, im0s)
                    return Output(file=Path(img_out), txt="")

                im0 = im0s.copy()
                txt_content = []

                det[:, :4] = scale_coords(self.current_img.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, conf, cls in det:
                    x1, y1, x2, y2 = [float(x.cpu().numpy()) for x in xyxy]
                    cls_id = int(cls)
                    conf_val = float(conf)
                    txt_content.append(f"{cls_id} {conf_val:.2f} {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f}")
                    plot_one_box(xyxy, im0, label=f"{self.names[cls_id]} {conf_val:.2f}", 
                               color=self.colors[cls_id])

                output_path = str(image).replace("/tmp", "", 1).replace("/", "")
                img_out = f"{out_dir}/{output_path}"
                txt_out = f"{out_dir}/{PathLib(output_path).stem}.txt"

                cv2.imwrite(img_out, im0)
                if txt_content:
                    with open(txt_out, 'w') as f:
                        f.write('\n'.join(txt_content))

                return Output(file=Path(img_out), txt='\n'.join(txt_content))

        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}")