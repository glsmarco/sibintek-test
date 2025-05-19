"""
Make folders with .json files with text-n-bboxes recognized on images of dataset.
One folder for english and one for cyrillic.
"""

from environment import Environment
from tqdm import tqdm
from image_processor import PaddleImageProcessor
import yaml

env = Environment()

input_images = [i for i in env.standartized_images_dir.iterdir()]

for language in env.ocr_languages:
    processor = PaddleImageProcessor(
        use_angle_cls=False,
        lang=language
    )
    dst_dir = env.standartized_images_dir.parent / language
    dst_dir.mkdir(exist_ok=True, parents=True)

    for input_image_path in tqdm(input_images, desc=f"running OCR <{language}> for dataset.."):
        ocr_result = processor.read_text_from_image(input_image_path)

        with open(dst_dir / input_image_path.with_suffix(".yaml").name, 'w', encoding="utf-8") as stream:
            yaml.safe_dump(ocr_result, stream, indent=2, allow_unicode=True)
