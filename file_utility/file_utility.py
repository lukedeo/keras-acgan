import os
import re

def is_file_extension(file_name, ext):
    return file_name.lower().endswith(ext)

def remove_current_dir_prefix(path):
    return path[:2].replace("./", "") + path[2:]

def read_all_files_path(root):
    paths = []
    for obj in os.listdir(root):
        path = root + "/" + obj
        if os.path.isfile(path):
            paths.append(path)
        elif os.path.isdir(path):
            paths += read_all_files_path(path)
    return paths

def read_all_files_absolute_path(root):
    return [os.path.abspath(path) for path in read_all_files_path(root)]

def read_child_files_name(root):
    return [obj for obj in os.listdir(root) if os.path.isfile(root + "/" + obj)]

def read_child_dirs_name(root):
    return [obj for obj in os.listdir(root) if os.path.isdir(root + "/" + obj)]
