import argparse
import yaml
from pathlib import Path
from PIL import Image, ImageDraw


def load_annotations(yaml_path: Path) -> list[dict]:
    """Load annotations from a YAML file."""
    with yaml_path.open('r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return data.get('ocr_results', [])


def draw_bboxes(image: Image.Image, annotations: list[dict], line_color: str = "red", line_width: int = 2) -> Image.Image:
    """Draw bounding boxes on the image."""
    draw = ImageDraw.Draw(image)
    for annotation in annotations:
        bbox = annotation.get('bbox', [])
        if bbox and all(isinstance(point, list) and len(point) == 2 for point in bbox):
            # Ensure points are tuples and close the polygon
            bbox_points = [tuple(point) for point in bbox] + [tuple(bbox[0])]
            draw.line(bbox_points, fill=line_color, width=line_width)
    return image


def main():
    parser = argparse.ArgumentParser(description="Draw OCR bounding boxes on an image.")
    parser.add_argument("--image_path", type=Path, required=True, help="Path to the .jpg image file.")
    parser.add_argument("--annotations_path", type=Path, required=True, help="Path to the YAML annotations file.")
    args = parser.parse_args()

    # Validate input files
    if not args.image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {args.image_path}")
    if not args.annotations_path.is_file():
        raise FileNotFoundError(f"Annotations file not found: {args.annotations_path}")

    # Load image and annotations
    image = Image.open(args.image_path).convert("RGB")
    annotations = load_annotations(args.annotations_path)

    # Draw and display
    image_with_bboxes = draw_bboxes(image, annotations)
    image_with_bboxes.show(title="Image with OCR Bounding Boxes")


if __name__ == "__main__":
    main()
