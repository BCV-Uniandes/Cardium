import json
import os
import shutil
import pathlib


def copy_images_by_trimester(dataset_path: str, trimester_json_path: str, output_dir: str):
    """
    Copy images from a structured dataset into separate directories organized by trimester.
    
    Args:
        base_dir (str or pathlib.Path): Path to the CARDIUM the dataset.
        trimester_json_path (str): Path to the JSON file containing trimester distribution.
        output_dir (str or pathlib.Path): Directory where the trimester-organized images will be copied.
    
    The function assumes the JSON has the following nested structure:
        trimester_distribution[trimester][fold][split][category][folder] -> list of image filenames
    
    Example hierarchy created in the output:
        output_dir/
            first_trimester/
                fold_1/
                    train/
                        CHD/
                            folder_id/
                                image1.jpg
                                image2.jpg
                        Non_CHD/
                    test/
                        ...
                fold_2/
                fold_3/
            second_trimester/
            third_trimester/
    """
    # Load the JSON containing trimester distribution
    with open(trimester_json_path, "r") as f:
        trimester_distribution = json.load(f)

    # Define constants
    trimesters = ["first_trimester", "second_trimester", "third_trimester"]
    folds = ["fold_1", "fold_2", "fold_3"]
    splits = ["train", "test"]
    categories = ["CHD", "Non_CHD"]
    
    # Iterate over trimesters, folds, splits, categories
    for trimester in trimesters:
        for fold in folds:
            for split in splits:
                for cat in categories:
                    folders = trimester_distribution[trimester][fold][split][cat]
                    for folder in folders:
                        src_dir = os.path.join(dataset_path, fold, split, cat, folder)
                        dst_dir = os.path.join(output_dir, trimester, fold, split, cat, folder)
                        os.makedirs(dst_dir, exist_ok=True)
                        
                        # Copy each image from source to destination
                        images = trimester_distribution[trimester][fold][split][cat][folder]
                        for image in images:
                            src_path = os.path.join(src_dir, image)
                            dst_path = os.path.join(dst_dir, image)
                            shutil.copy(src_path, dst_path)


BASE_DIR = pathlib.Path(__name__).resolve().parent.parent
dataset_path = os.path.join(BASE_DIR, "dataset/cardium_images")
trimester_path = os.path.join(BASE_DIR, "trimester_results/trimester_distribution.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "dataset/trimester_images")

copy_images_by_trimester(dataset_path=dataset_path, trimester_json_path=trimester_path, output_dir=OUTPUT_DIR)