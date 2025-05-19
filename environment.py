from pathlib import Path
import yaml
from dataclasses import dataclass
from functools import cached_property
from benedict import benedict
from typing import List


@dataclass
class Environment:

    @cached_property
    def base_path(self) -> Path:
        return Path(__file__).resolve().parent
    
    @cached_property
    def config_path(self) -> Path:
        result: Path = self.base_path / "config.yaml"
        if not result.is_file():
            raise FileNotFoundError("config file not found at project / config.yaml")
        return result
    
    @cached_property
    def config(self) -> benedict:
        return benedict(
            yaml.safe_load(
                Path(self.config_path).read_text()
            )
        )
    
    @cached_property
    def raw_images_dir(self) -> Path:
        return self.base_path / self.config.raw_images_dir
    
    @cached_property
    def standartized_images_dir(self) -> Path:
        return self.base_path / self.config.standartized_images_dir
    
    @cached_property
    def ocr_languages(self) -> List[str]:
        langs = self.config.ocr_languages
        if langs is None:
            return []
        return langs
    
    @cached_property
    def target_size(self) -> int:
        return self.config.target_size
    
    @cached_property
    def gt_annotations_path(self) -> Path:
        return self.standartized_images_dir.parent / "gt_annotations.yaml"
