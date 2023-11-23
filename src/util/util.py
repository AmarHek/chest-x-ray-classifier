import os
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
