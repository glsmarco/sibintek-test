"""
this module provides methods to process annotations pre-made by PaddleOCR 
"""
from dataclasses import dataclass
from typing import List, Sequence, Optional, Tuple, Generator
from pathlib import Path
import yaml
from environment import Environment
import math
import re
from itertools import product


ID_LENGTH = 10


@dataclass
class Point:

    x: float
    y: float

    def distance(self, other: 'Point') -> float:
        return math.hypot(
            self.x - other.x,
            self.y - other.y
        )

@dataclass
class Bbox:
    
    tl: Point
    tr: Point
    br: Point
    bl: Point

    @property
    def points(self) -> Sequence[Point]:
        return self.tl, self.tr, self.br, self.bl

    @property
    def x_min(self) -> float:
        """
        this is because I am not quite sure that PaddleOCR polygon's tl is real top left as it is not a rectangle
        """
        return min(p.x for p in self.points)

    @property
    def x_max(self) -> float:
        return max(p.x for p in self.points)    
    
    @property
    def y_min(self) -> float:
        return min(p.y for p in self.points)

    @property
    def y_max(self) -> float:
        return max(p.y for p in self.points)   

    @property
    def center(self) -> Point:
        # average of all four corners
        xs = [p.x for p in self.points]
        ys = [p.y for p in self.points]
        return Point(sum(xs) / 4, sum(ys) / 4)

    @property
    def y_range(self) -> Tuple[float, float]:
        return self.y_min, self.y_max
    
    @property
    def height(self) -> float:
        return self.y_range[1] - self.y_range[0]

    @property
    def x_range(self) -> Tuple[float, float]:
        return self.x_min, self.x_max

    @property
    def width(self) -> float:
        return self.x_range[1] - self.x_range[0]

@dataclass
class Prediction:

    bbox: Bbox
    confidence: float
    text: str



class PredictionProcessor:

    """
    Processes Paddle OCR predictions to extract vehicle passport IDs from document images
    """

    ID_PATTERN = re.compile(r"^\d{2}[A-Za-z]{2}\d{6}$")
    LETTER_CORRECTIONS = {
        "V": "Y",  # sometimes “Y” is read as “V” and "V" could not be in the pattern at all
        "v": "Y",
    }

    def __init__(self, env: Environment):
        self.target_size: int = env.target_size
        self.loaded_predictions: Optional[Sequence[Prediction]] = None

    def load_predictions(self, predictions_path: str) -> None:
        """
        load predictions from annotation file
        in-place modifies self.loaded_predictions field
        """
        
        predictions: List[Prediction] = self.read_predictions_from_yaml(predictions_path)
        self.loaded_predictions = predictions

    def read_predictions_from_yaml(self, path: str) -> List[Prediction]:
        res = []
        data = yaml.safe_load(Path(path).read_text())["ocr_results"]
        for item in data:
            res.append(
                Prediction(
                    bbox=Bbox(*[Point(*i) for i in item["bbox"]]),
                    confidence=item["confidence"],
                    text=item["text"]
                )
            )
        return res
    
    def sort_by_vicinity_to_tr(self) -> None:
        """
        in-place modifies self.loaded_predictions
        sort predictions by their vicinity to top right corner
        this is because I chose couple of text objects from the right side of a document
        as in most files this side is present in normal quality
        """
        self._unloaded_predictions_check()

        top_right = Point(self.target_size, 0)
        self.loaded_predictions = sorted(
            self.loaded_predictions, 
            key=lambda p: p.bbox.center.distance(top_right)
        )


    def _unloaded_predictions_check(self) -> None:
        """
        @raises RuntimeError if predictions are not loaded yet
        """
        if self.loaded_predictions is None:
            raise RuntimeError(f"no predictions are loaded. Please run load_predictions(path-predictions) first")
        

    @staticmethod
    def ensure_loaded_and_tr_sorted(method):
        def wrapper(self, *args, **kwargs):
            self._unloaded_predictions_check()
            self.sort_by_vicinity_to_tr()
            return method(self, *args, **kwargs)
        return wrapper
    
    @ensure_loaded_and_tr_sorted
    def read_vehicle_passport_id(self) -> Optional[str]:
        
        for prediction in self.loaded_predictions:
            text = prediction.text.replace(" ", "")
            corrected_= self._matches_id(text)
            if corrected_:
                return corrected_
            if text[:2].isdigit():
                result = self._try_build_full_id(prediction, self.loaded_predictions)
                if result:
                    return result
        return None                

    def _id_variants(self, candidate: str) -> Generator[str, None, None]:
        """
        For a 10-char string, generate all variants where positions 2 and 3
        may be corrected/swapped:
          - if OCR gave '0' or 'O', allow both
          - if OCR gave a letter in LETTER_CORRECTIONS, replace with the mapped letter
        """
        candidate = candidate.upper()
        if len(candidate) != ID_LENGTH:
            return []
        chars = list(candidate)
        # the two letter positions in the pattern
        idxs = (2, 3)

        slot_options = []
        for i in idxs:
            c = chars[i]
            opts: list[str] = []

            # 1) if there’s a V→Y rule, try Y first
            if corr := self.LETTER_CORRECTIONS.get(c):
                opts.append(corr)

            # 2) if it’s O/0 ambiguity, try 'O' then '0'
            if c in ("O", "0"):
                opts.extend(["O", "0"])
            else:
                # 3) finally try the original OCR’d char
                opts.append(c)

            slot_options.append(opts)

        # cartesian product of the two slots
        for ch2, ch3 in product(*slot_options):
            v = chars.copy()
            v[2], v[3] = ch2, ch3
            yield "".join(v)

    def _matches_id(self, candidate: str) -> Optional[str]:
        """
        Return the first corrected variant that matches, or None.
        """
        for variant in self._id_variants(candidate):
            if self.ID_PATTERN.fullmatch(variant):
                return variant
        return None

    def _try_build_full_id(self, start: Prediction, all_preds: List[Prediction]) -> Optional[str]:
        """
        Given a starting prediction, find all to its right on roughly the same line,
        concatenate in nearest‐first order, and check for full‐ID match.
        """
        y_min, y_max = start.bbox.y_range
        cx = start.bbox.center.x

        # candidates: to the right and overlapping y-range
        neighbors = [
            p for p in all_preds
            if p is not start
            and p.bbox.center.x > cx
            and y_min <= p.bbox.center.y <= y_max
        ]
        # sort by horizontal distance from start
        neighbors.sort(key=lambda p: p.bbox.center.x - cx)

        # try concatenations of increasing length
        texts = [start.text.replace(" ", "")]
        for nb in neighbors:
            texts.append(nb.text.replace(" ", ""))
            candidate = "".join(texts)
            corrected = self._matches_id(candidate)
            if corrected:
                return corrected
            if len(candidate) > ID_LENGTH: # length of this pattern
                break
        return None
           




if __name__ == "__main__":
    pass

    