def get_file_contents_as_list(filename):
    with open(filename, 'r') as file:
        lines = [line.rstrip() for line in file]
    return lines


def serialize_numpy_array(numpy_array):
    serialized_dict = {}

    for idx, output in enumerate(numpy_array):
        serialized_dict[idx] = output.tolist()

    return serialized_dict
