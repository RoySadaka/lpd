import os
import pickle
import gzip
from shutil import copytree, copyfile, move

def is_folder_exists(folder_path):
    return os.path.exists(folder_path)

def create_folder(full_path):
    os.makedirs(full_path)

def delete_file(full_path):
    os.remove(full_path)

def copy_paste_file(source_path, dest_path):
    copyfile(source_path, dest_path)

def copy_paste_folder(source_path, dest_path, only_content = False):
    if only_content:
        for root, dirs, files in os.walk(source_path):
            for file_name in files:
                copyfile(source_path + file_name, dest_path + file_name)
            for dir_name in dirs:
                copytree(source_path + dir_name, dest_path)
    else:
        #WHOLE DIRECTORY
        copytree(source_path, dest_path)

def move_folder_location(source_path, dest_path):
    move(source_path, dest_path)

def move_file_location(source_path, dest_path):
    move(source_path, dest_path)

def load_object_from_file(file_path, verbose = 1):
    with_pickle_suffix = file_path + '.pickle'
    if verbose == 1:
        print('loading object from file:', with_pickle_suffix)
    with open(with_pickle_suffix, 'rb') as f:
        obj = pickle.load(f)
    return obj

def save_object_as_file(obj, file_path):
    with_pickle_suffix = file_path + '.pickle'
    print('saving object to file:', with_pickle_suffix)
    with open(with_pickle_suffix, 'wb') as file:
        pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)

def load_object_from_compressed_file(file_path, verbose = 1):
    with_pickle_suffix = file_path + '.pickle'
    if verbose == 1:
        print('loading object from compressed file:', with_pickle_suffix)
    with gzip.GzipFile(with_pickle_suffix, 'r') as f:
        obj = pickle.load(f)
    return obj

def save_object_as_compressed_file(obj, file_path):
    with_pickle_suffix = file_path + '.pickle'
    print('saving object to compressed file:', with_pickle_suffix)
    with gzip.GzipFile(with_pickle_suffix, 'w') as file:
        pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)

def get_num_files_in_directory(directory, prefix = None, suffix = None):
    count = 0
    for _, _, files in os.walk(directory):
        for file_name in files:
            if prefix is not None:
                if not file_name.startswith(prefix):
                    continue
            
            if suffix is not None:
                if not file_name.endswith(suffix):
                    continue
                
            count += 1
    return count

def write_text_to_file(text, full_path):
    text_file = open(f"{full_path}.txt", "w")
    text_file.write(text)
    text_file.close()

def read_all_text_from_file(full_path):
    file = open(full_path,mode='r')
    all_text = file.read()
    file.close() 
    return all_text

def loop_file_names_in_directory(root):
    for dir_path, dir_names, file_names in os.walk(root):
        for file_name in file_names:
            full_file_path = dir_path + os.sep + file_name
            yield full_file_path
