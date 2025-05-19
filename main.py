"""
This script runs annotation evaluation logic. 
Production of annotations was handled in separate script: produce_ocr_annotations.py. 
This is because my local setup does not meet PaddleOCR CUDA requirements out of the box.
"""
from environment import Environment
import yaml
from annotation_processor import PredictionProcessor
from pathlib import Path


def run_id_evaluation():
    env = Environment()
    gt_annotations = yaml.safe_load(env.gt_annotations_path.read_text())

    id_total = 0
    id_correct = 0
    for idx, sample in enumerate(sorted(env.standartized_images_dir.iterdir(), key=lambda x: int(x.stem.split("_")[1]))): # sorting is just cosmetic here
        if idx == 0: 
            continue
        ## id reading
        id_total += 1
        corresponding_prediction = Path(f"dataset/en/{sample.with_suffix('.yaml').name}").resolve()
        processor = PredictionProcessor(env)
        processor.load_predictions(predictions_path=str(corresponding_prediction))
        id = processor.read_vehicle_passport_id()
        if id is not None:
            id = id[:2] + " " + id[2:4] + " " + id[4:]
            correct_id = gt_annotations[sample.name]['id']
            if id == correct_id:
                id_correct += 1
        

        print(f"{idx} : {sample.name}")
        print(f"    id: {id} ::: correct_id: {correct_id} ::: id match: {id == correct_id}")
    print(f"id reading accuracy: {100 * id_correct / id_total} %")

if __name__ == "__main__":
    run_id_evaluation()