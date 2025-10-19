import argparse
from ultralytics import YOLO  # type: ignore

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a YOLOv11 model on a custom dataset."
    )
    parser.add_argument(
        "-n", "--name",
        type=str,
        default="yolov11_custom_train",
        help="Name for the training run.",
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="yolo11n.pt",
        help="Path to the YOLOv11 model weights.",
    )
    parser.add_argument(
        "-d", "--data",
        type=str,
        required=True,
        help="Path to the dataset YAML file.",
    )
    parser.add_argument(
        "-e", "--epochs",
        type=int,
        default=300,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "-b", "--batch-size",
        type=int,
        default=32,
        help="Batch size for training.",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    model = YOLO(args.model)
    
    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=120,
        name=args.name,
    )