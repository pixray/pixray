import os

def create_output_file(outfile_path, suffix, content):
    file_name = get_output_file_name(outfile_path, suffix)

    if not file_name: return

    file = open(file_name, 'w+')
    file.write(content)

# Note: Eventually this should be replaced by time of generation + seed
def get_output_file_name(outfile_path, suffix):
    file_name, _ = os.path.splitext(outfile_path)
    if not (file_name and file_name.strip()): return None

    return file_name + suffix
