import logging
import argparse
import random
import threading
from pathlib import Path
from typing import Dict, Tuple
from PIL.Image import Image
from PIL import Image as PILImage
import numpy as np

from _types import ImageSize, Variants

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATASET_PATH = Path(__file__).parent / "datasets"
RAW_IMAGE_SIZE = ImageSize(width=97, height=120)
NORMALIZED_IMAGE_SIZE = ImageSize(width=120, height=120)
IMAGE_VARIANTS = [
    Variants.GRAYSCALE,
    Variants.CONTRAST,
    Variants.BRIGHT,
    Variants.COMBINED,
    Variants.ORIGINAL,
]


def get_random_color() -> Tuple[int, int, int]:
    """Generate a random RGB color."""
    return (
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255),
    )

def letterbox_image(image_path: Path, target_size: ImageSize) -> Image:
    """Resize image with unchanged aspect ratio using padding."""
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    tgt_w, tgt_h = int(target_size.width), int(target_size.height)

    with PILImage.open(image_path) as img:
        img_rgb = img.convert("RGB")
        src_w, src_h = img_rgb.size

        # scale to fit target preserving aspect ratio
        scale = min(tgt_w / src_w, tgt_h / src_h)
        new_w = max(1, int(round(src_w * scale)))
        new_h = max(1, int(round(src_h * scale)))

        img_resized = img_rgb.resize((new_w, new_h), PILImage.Resampling.LANCZOS)

        fill = (114, 114, 114)  # neutral padding color
        canvas = PILImage.new("RGB", (tgt_w, tgt_h), fill)

        # center on canvas
        x = (tgt_w - new_w) // 2
        y = (tgt_h - new_h) // 2
        canvas.paste(img_resized, (x, y))

        return canvas


def normalize_string(s: str) -> str:
    return s.strip().lower().replace(" ", "_")


def infer_label_name(img_path: Path) -> str:
    """Infer label name from image path by checking prefix or parent folder."""
    stem = normalize_string(img_path.stem)
    if "_" in stem:
        return stem.split("_", 1)[0]
    parent = normalize_string(img_path.parent.name)
    return parent if parent else stem


def build_label_map(raw_data_path: Path, use_fullname: bool = True) -> Dict[str, int]:
    """Build label map from raw data directory."""
    image_paths = list(raw_data_path.rglob("*.jpg")) + list(
        raw_data_path.rglob("*.png")
    )
    names = set()
    for p in image_paths:
        if use_fullname:
            names.add(normalize_string(p.stem))
        else:
            names.add(infer_label_name(p))
    sorted_names = sorted(
        names, key=lambda s: (s.isdigit(), s)
    )  # stable ordering; non-digits first
    return {name: idx for idx, name in enumerate(sorted_names)}


def resolve_class_name(img_path: Path, label_map: Dict[str, int]) -> str:
    """Resolve class name for an image using label_map."""
    candidates = [
        normalize_string(img_path.stem),  # full image name
        infer_label_name(img_path),  # prefix or parent folder
        normalize_string(img_path.parent.name),  # parent folder
    ]
    for c in candidates:
        if c in label_map:
            return c
    raise KeyError(f"Could not resolve class for {img_path}. Tried: {candidates}")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def worker_task(
    paths: list[Path],
    worker_id: int,
    output_dir: Path,
    label_map: Dict[str, int],
    variants: list[Variants] = IMAGE_VARIANTS,
    instances_per_variant: int = 1,
) -> None:
    """Worker task to process images and generate variants."""
    # variants may be passed positionally from threaded args; ensure it's iterable
    if not hasattr(variants, "__iter__") or isinstance(variants, (int, float)):
        variants = IMAGE_VARIANTS
    for img_path in paths:
        try:
            img_out_path = output_dir / "images" / "train"
            label_out_path = output_dir / "labels" / "train"

            ensure_dir(img_out_path)
            ensure_dir(label_out_path)

            # open source card
            try:
                with PILImage.open(img_path) as src:
                    card = src.convert("RGBA")
            except Exception as e_open:
                logger.warning(f"Worker {worker_id} could not open image {img_path}: {e_open}")
                continue

            # determine class id (resolve using image name first)
            try:
                class_name = resolve_class_name(img_path, label_map)
            except KeyError:
                logger.warning(
                    f"Unknown class for image '{img_path}'; candidates not in label map; skipping"
                )
                continue
            class_id = label_map[class_name]

            cw = int(NORMALIZED_IMAGE_SIZE.width)
            ch = int(NORMALIZED_IMAGE_SIZE.height)

            # helper to place an RGBA image onto canvas at random position/rotation and save label
            def place_and_save(stem: str, suffix: str, variant_img: PILImage.Image):
                # scale the variant image randomly (keeps some variability)
                scale = random.uniform(0.3, 0.95)
                nw = max(1, int(round(variant_img.width * scale)))
                nh = max(1, int(round(variant_img.height * scale)))
                if nw > cw or nh > ch:
                    fit_scale = min(cw / variant_img.width, ch / variant_img.height) * 0.95
                    nw = max(1, int(round(variant_img.width * fit_scale)))
                    nh = max(1, int(round(variant_img.height * fit_scale)))
                placed = variant_img.resize((nw, nh), PILImage.Resampling.LANCZOS)

                # optional small rotation
                if random.random() < 0.3:
                    placed = placed.rotate(random.uniform(-15, 15), expand=True)

                nw, nh = placed.size
                if nw >= cw:
                    nw = cw - 1
                if nh >= ch:
                    nh = ch - 1

                max_x = cw - nw
                max_y = ch - nh
                if max_x <= 0 or max_y <= 0:
                    x = 0
                    y = 0
                else:
                    x = random.randint(0, max_x)
                    y = random.randint(0, max_y)

                canvas = PILImage.new("RGB", (cw, ch), get_random_color())
                paste_img = placed.convert("RGBA")
                canvas.paste(paste_img, (x, y), paste_img)

                out_fname = f"{stem}{suffix}{img_path.suffix}"
                out_img_path = img_out_path / out_fname
                canvas.save(out_img_path, quality=90)

                # compute bbox and write label in YOLO format
                x_min, y_min = x, y
                x_max, y_max = x + nw, y + nh
                cx = (x_min + x_max) / 2.0 / cw
                cy = (y_min + y_max) / 2.0 / ch
                w_norm = (x_max - x_min) / cw
                h_norm = (y_max - y_min) / ch

                lbl_path = label_out_path / f"{stem}{suffix}.txt"
                lbl_path.write_text(f"{class_id} {cx:.6f} {cy:.6f} {w_norm:.6f} {h_norm:.6f}\n")

            stem = img_path.stem

            # create and save each requested variant using random placement
            def place_multiple(stem: str, base_suffix: str, variant_img: PILImage.Image):
                for inst_idx in range(instances_per_variant):
                    place_and_save(stem, f"{base_suffix}_{inst_idx}", variant_img)

            if Variants.ORIGINAL in variants:
                place_multiple(stem, "_original", card)
            if Variants.BRIGHT in variants:
                bright = PILImage.eval(card.convert("RGB"), lambda v: min(255, int(v * 1.2))).convert("RGBA")
                place_multiple(stem, "_bright", bright)
            if Variants.GRAYSCALE in variants:
                gray = card.convert("L").convert("RGBA")
                place_multiple(stem, "_grayscale", gray)
            if Variants.CONTRAST in variants:
                contrast = PILImage.eval(card.convert("RGB"), lambda v: min(255, max(0, int(128 + 1.5 * (v - 128))))).convert("RGBA")
                place_multiple(stem, "_contrast", contrast)
            if Variants.COMBINED in variants:
                combined = PILImage.eval(card.convert("RGB"), lambda v: min(255, max(0, int(128 + 1.5 * (v - 128) * 1.2)))).convert("RGBA")
                place_multiple(stem, "_combined", combined)

            logger.info(f"Worker {worker_id} processed {img_path.stem} successfully.")

        except Exception as e:
            logger.error(f"Worker {worker_id} failed to process {img_path}: {e}")


def build_data_yaml(
    output_dir: Path,
    num_classes: int,
    class_names: list[str],
    train_subdir: str = "train",
    val_subdir: str = "val",
) -> None:
    """Builds a data.yaml file for the dataset."""
    if num_classes != len(class_names):
        logger.warning(
            "num_classes (%d) does not match length of class_names (%d). Using len(class_names).",
            num_classes,
            len(class_names),
        )
        num_classes = len(class_names)

    train_path = (output_dir / "images" / train_subdir).as_posix()
    val_path = (output_dir / "images" / val_subdir).as_posix()
    names_list = ", ".join(f"'{normalize_string(n)}'" for n in class_names)

    yaml_content = (
        f"train: {train_path}\n"
        f"val:   {val_path}\n\n"
        f"nc: {num_classes}\n"
        f"names: [{names_list}]\n"
    )

    yaml_path = output_dir / "data.yaml"
    yaml_path.write_text(yaml_content)
    logger.info("Wrote dataset yaml to %s", yaml_path)


def generate_synthetic_val(
    raw_data_path: Path,
    output_dir: Path,
    label_map: Dict[str, int],
    n_samples: int = 200,
    canvas_size: tuple[int, int] = (
        int(NORMALIZED_IMAGE_SIZE.width),
        int(NORMALIZED_IMAGE_SIZE.height),
    ),
    scale_range: tuple[float, float] = (0.3, 0.9),
    use_real_bgs: Path | None = None,
) -> int:
    """Generate synthetic validation samples by pasting card images onto random or real backgrounds."""
    ensure_dir(output_dir / "images" / "val")
    ensure_dir(output_dir / "labels" / "val")

    # collect card images
    cards = [p for p in raw_data_path.rglob("*.jpg")] + [
        p for p in raw_data_path.rglob("*.png")
    ]
    if not cards:
        logger.warning(
            "No card images found in %s, skipping synthetic generation.", raw_data_path
        )
        return 0

    real_bgs = []
    if use_real_bgs and use_real_bgs.exists():
        real_bgs = [p for p in use_real_bgs.rglob("*.jpg")] + [
            p for p in use_real_bgs.rglob("*.png")
        ]

    cw, ch = int(canvas_size[0]), int(canvas_size[1])
    written = 0

    for i in range(n_samples):
        try:
            # choose background
            if real_bgs and random.random() < 0.6:
                bg_path = random.choice(real_bgs)
                with PILImage.open(bg_path) as bg_img:
                    bg = bg_img.convert("RGB")
                    bg = (
                        PILImage.Image.resize(bg, (cw, ch))
                        if hasattr(PILImage.Image, "resize")
                        else bg.resize((cw, ch))
                    )
                    bg = PILImage.new("RGB", (cw, ch))
            else:
                bg = PILImage.new("RGB", (cw, ch), (114, 114, 114))

            # start canvas and paste background
            canvas_rgb = PILImage.new("RGB", (cw, ch), (114, 114, 114))
            canvas_rgb.paste(bg, (0, 0))

            labels: list[str] = []
            placed_any = False

            card_path = random.choice(cards)
            try:
                with PILImage.open(card_path) as cimg:
                    card = cimg.convert("RGBA")

                    # scale to fit canvas if needed
                    scale = random.uniform(*scale_range)
                    nw = max(1, int(round(card.width * scale)))
                    nh = max(1, int(round(card.height * scale)))
                    if nw > cw or nh > ch:
                        fit_scale = min(cw / card.width, ch / card.height) * 0.95
                        nw = max(1, int(round(card.width * fit_scale)))
                        nh = max(1, int(round(card.height * fit_scale)))
                    card_resized = card.resize(
                        (nw, nh), PILImage.Resampling.LANCZOS
                    )

                    # optional small rotation
                    if random.random() < 0.3:
                        card_resized = card_resized.rotate(
                            random.uniform(-10, 10), expand=True
                        )

                    nw, nh = card_resized.size
                    # ensure fits, recompute dims
                    if nw >= cw:
                        nw = cw - 1
                    if nh >= ch:
                        nh = ch - 1

                    max_x = cw - nw
                    max_y = ch - nh
                    if max_x <= 0 or max_y <= 0:
                        x = 0
                        y = 0
                    else:
                        x = random.randint(0, max_x)
                        y = random.randint(0, max_y)

                    # paste using alpha if present
                    paste_img = card_resized.convert("RGBA")
                    canvas_rgb.paste(paste_img.convert("RGBA"), (x, y), paste_img)

                    # classification / detection label id
                    try:
                        classname = resolve_class_name(card_path, label_map)
                    except KeyError:
                        # fallback to infer_label_name
                        classname = infer_label_name(card_path)
                        if classname not in label_map:
                            logger.debug(
                                "Skipping card %s; class %s not in label_map",
                                card_path,
                                classname,
                            )
                            continue
                    class_id = label_map[classname]

                    # bbox in pixels
                    x_min, y_min = x, y
                    x_max, y_max = x + nw, y + nh

                    # normalized YOLO format
                    cx = (x_min + x_max) / 2.0 / cw
                    cy = (y_min + y_max) / 2.0 / ch
                    w_norm = (x_max - x_min) / cw
                    h_norm = (y_max - y_min) / ch

                    labels.append(
                        f"{class_id} {cx:.6f} {cy:.6f} {w_norm:.6f} {h_norm:.6f}"
                    )
                    placed_any = True

            except Exception as ex_card:
                logger.debug(
                    "Failed to place card %s in sample %d: %s",
                    card_path,
                    i,
                    ex_card,
                )
                continue

            if not placed_any:
                # nothing placed, skip writing this sample
                continue

            fname = f"syn_val_{i:06d}.jpg"
            img_path = output_dir / "images" / "val" / fname
            lbl_path = output_dir / "labels" / "val" / (Path(fname).stem + ".txt")

            canvas_rgb.save(img_path, quality=90)
            lbl_path.write_text("\n".join(labels) + "\n")

            written += 1

        except Exception as ex:
            logger.debug("Failed to create synthetic sample %d: %s", i, ex)
            continue

    logger.info(
        "Generated %d synthetic validation samples at %s", written, output_dir / "val"
    )
    return written


def build_dataset(
    dataset_name: str,
    raw_data_path: Path,
    output_dir: Path,
    num_workers: int = 4,
    instances_per_variant: int = 1,
) -> None:
    """Build dataset from raw images located in raw_data_path and save to output_dir."""
    logger.info(f"Building dataset '{dataset_name}' from raw data at {raw_data_path}")

    label_map = build_label_map(raw_data_path)
    classes_path = output_dir / "classes.txt"
    classes_path.write_text(
        "\n".join(name for name, _ in sorted(label_map.items(), key=lambda kv: kv[1]))
    )
    logger.info(f"Wrote classes file to {classes_path} with {len(label_map)} classes")

    load = list(raw_data_path.rglob("*.jpg")) + list(raw_data_path.rglob("*.png"))
    balanced_load = balance_load(load, num_workers)

    ensure_dir(output_dir / "images" / "train")
    ensure_dir(output_dir / "labels" / "train")

    worker_threads = []
    for worker_idx, image_paths in enumerate(balanced_load):
        logger.info(f"Worker {worker_idx} processing {len(image_paths)} images")
        thread = threading.Thread(
            target=worker_task,
            args=(image_paths, worker_idx + 1, output_dir, label_map),
            kwargs={"instances_per_variant": instances_per_variant},
        )
        thread.start()
        worker_threads.append(thread)

    for thread in worker_threads:
        logger.debug(f"Waiting for worker thread {thread.name} to finish")
        thread.join()

    logger.info("All worker threads have completed processing.")

    build_data_yaml(
        output_dir,
        num_classes=len(label_map),
        class_names=[
            name for name, _ in sorted(label_map.items(), key=lambda kv: kv[1])
        ],
    )

    logger.info(f"Data YAML file created at {output_dir / 'data.yaml'}")

    generate_synthetic_val(
        raw_data_path,
        output_dir,
        label_map,
        n_samples=200,
    )

    logger.info(f"Synthetic validation data generated at {output_dir / 'val'}")

    logger.info(f"Dataset '{dataset_name}' built successfully at {output_dir}")


def balance_load(image_list: list[Path], num_workers: int) -> list[list[Path]]:
    """Distribute images evenly across workers for balanced processing."""
    balanced = [[] for _ in range(num_workers)]
    for idx, image_path in enumerate(image_list):
        worker_idx = idx % num_workers
        balanced[worker_idx].append(image_path)
    return balanced


def parse_args():
    parser = argparse.ArgumentParser(description="Build dataset for SICrMLB models.")
    parser.add_argument(
        "-dn",
        "--dataset-name",
        type=str,
        required=True,
        help="Name of the dataset to build.",
    )
    parser.add_argument(
        "-rd",
        "--raw-data-path",
        type=Path,
        required=True,
        help="Path to the raw data directory.",
    )
    parser.add_argument(
        "-od",
        "--output-dir",
        type=Path,
        default=DATASET_PATH,
        help="Directory to save the processed dataset.",
    )
    parser.add_argument(
        "-nw",
        "--num-workers",
        type=int,
        default=4,
        help="Number of worker threads for data processing.",
    )
    parser.add_argument(
        "-ipc",
        "--instances-per-variant",
        type=int,
        default=10,
        help="Number of instances to generate per variant per image.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    output_dir = args.output_dir / args.dataset_name

    ensure_dir(output_dir)
    if not args.raw_data_path.exists():
        logger.error(f"Raw data path does not exist: {args.raw_data_path}")
        raise FileNotFoundError(f"Raw data path does not exist: {args.raw_data_path}")

    build_dataset(
        args.dataset_name,
        args.raw_data_path,
        output_dir,
        num_workers=args.num_workers,
        instances_per_variant=args.instances_per_variant,
    )
