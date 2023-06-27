def get_file_contents_as_list(filename):
    with open(filename, 'r') as file:
        lines = [line.rstrip() for line in file]
    return lines
