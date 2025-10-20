import torch
import argparse
from ultralytics import YOLO  # type: ignore

def parse_args():
    parser = argparse.ArgumentParser(
        description="Test a YOLOv11 model"
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="yolo11n.pt",
        help="Path to the YOLOv11 model weights.",
    )
    parser.add_argument(
        "-i", "--image",
        type=str,
        required=True,
        help="Path to the image to test.",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    model = YOLO(args.model)
    
    if torch.cuda.is_available():
        print("Using CUDA for testing.")
        device = "cuda"
    else:
        print("Using CPU for testing.")
        device = "cpu"
    
    results = model.predict(
        source=args.image,
        imgsz=120,
        device=device,
    )
    
    for result in results:
        result.show()