from paddleocr import PaddleOCR
from pathlib import Path
from interfaces import ImageProcessorInterface
from functools import cached_property
from typing import Union


class PaddleImageProcessor(ImageProcessorInterface):

    def __init__(
            self,
            use_angle_cls: bool=True, # helps autocorrect text orientation before recognition with cost of extra processing time
            lang: str="en" 
        ):
        super().__init__() # in case interface will start to implement stuff

        self.use_angle_cls = use_angle_cls
        self.lang = lang
    
    @cached_property
    def ocr(self) -> PaddleOCR:
        return PaddleOCR(
            use_angle_cls=self.use_angle_cls,
            lang=self.lang
        )
    
    def read_text_from_image(self, image_path: Union[str, Path]) -> dict:
        """
        @param image_path: Union[str, Path]
        @returns dict: OCR results with detected texts and their bounding boxes
        @raises FileNotFoundError if image_path not found
        """
        image_path = Path(image_path).resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        ocr_result = self.ocr.ocr(str(image_path), cls=self.use_angle_cls)

        # Convert output to structured dictionary
        results_dict_formatted = []
        for line in ocr_result[0]:  # First page (always index 0 for images)
            bbox, (text, confidence) = line
            results_dict_formatted.append({
                "text": text,
                "confidence": round(confidence, 4),
                "bbox": [list(map(int, point)) for point in bbox]
            })

        return {"ocr_results": results_dict_formatted}
    

if __name__ == "__main__":
    pass