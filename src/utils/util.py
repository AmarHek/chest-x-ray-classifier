import argparse
import os

import yaml
from tqdm import tqdm
from multiprocessing import Pool


def get_file_contents_as_list(filename):
    with open(filename, 'r') as file:
        lines = [line.rstrip() for line in file]
    return lines


def serialize_numpy_array(numpy_array):
    serialized_dict = {}

    for idx, output in enumerate(numpy_array):
        serialized_dict[idx] = output.tolist()

    return serialized_dict


def is_file(path):
    return os.path.isfile(path)


def check_files(paths):
    with Pool() as pool, tqdm(total=len(paths), desc="Checking files", unit="file") as pbar:
        results = list(tqdm(pool.imap(is_file, paths), total=len(paths), desc="Checking files", unit="file", leave=False))

    return results


def uniquify(path):
    # check if path or file, throw exception if file
    if os.path.isfile(path):
        raise ValueError("Path is a file, not a directory!")

    if os.path.exists(path):
        print(f"{path} already exists! Trying to find unique filename...")

    counter = 1
    new_path = path

    while os.path.exists(new_path):
        new_path = path + "_" + str(counter)
        counter += 1

        if counter > 10000:
            raise RuntimeError("Too many folders with the same name!")

    if new_path != path:
        print(f"Found unique path: {new_path}")

    return new_path


def tuple_constructor(loader, node):
    return tuple(loader.construct_sequence(node))


def load_yaml(yaml_path: os.PathLike):
    # Add the tuple constructor to the yaml loader
    yaml.add_constructor('tag:yaml.org,2002:python/tuple', tuple_constructor)
    with open(yaml_path, 'r') as file:
        contents = yaml.load(file, Loader=yaml.FullLoader)
    return contents


def dir_path(path):
    if os.path.exists(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")
