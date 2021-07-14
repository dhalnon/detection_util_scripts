import os
import numpy as np
import argparse
import shutil

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', metavar='d', type=str)
    parser.add_argument('train_percentage', metavar='t', type=int)
    return parser.parse_args()


def make_dirs(directory):
    val_dir = os.path.join(directory, 'val')
    val_dirs = {
        'base'          :val_dir,
        'annotations'   :os.path.join(val_dir, 'annotations'),
        'images'        :os.path.join(val_dir, 'images')
    }

    train_dir = os.path.join(directory, 'train')
    train_dirs = {
        'base'          :train_dir,
        'annotations'   :os.path.join(train_dir, 'annotations'),
        'images'        :os.path.join(train_dir, 'images')
    }

    dir_list = [*train_dirs.values(), *val_dirs.values()]

    for dir in list(dir_list):
        os.makedirs(dir, exist_ok=True)
    return val_dirs, train_dirs


def get_files_list(directory):
    files_list = []
    for root, _, files in os.walk(directory):
        for f in files:
            if f.endswith('.xml'):
                files_list.append(os.path.join(root, f))
    return files_list


def split_decision(train_percentage):
    if np.random.randint(1, 101) > train_percentage:
        return False

    return True


def generate_savepaths(file, anno_dir, image_dir):
    filename = file.split('/')[-1]
    image_filename = filename.replace('.xml', '')
    anno_path = os.path.join(anno_dir, filename)
    image_path = os.path.join(image_dir, image_filename)
    return anno_path, image_path


def move_files(src_anno, src_image, dst_anno, dst_image):
    shutil.move(src_anno, dst_anno)
    shutil.move(src_image, dst_image)
    

def link_files(files_list):
    linked_files = {}

    i = 0
    for file in files_list:
        print(i, end='\r')
        image_filepath = file.replace('annotations','images').replace('.xml','')
        image_filename = image_filepath.split(os.sep)[-1]
        anno_filename = file.split(os.sep)[-1]
        companions = {
            'image_filepath' : image_filepath,
            'annotation_filepath' : file,
            'image_filename' : image_filename,
            'annotation_filename'  : anno_filename
        }
        linked_files[i] = companions
        i += 1
    return linked_files


def main():
    args = parse_args()
    
    val_dirs, train_dirs = make_dirs(args.directory)

    files_list = get_files_list(args.directory)
    
    linked_files = link_files(files_list)
    print('\n')
    i = 0
    for key in linked_files.keys():
        print(i, end='\r')
        current_anno_path = linked_files[key]['annotation_filepath']
        current_image_path = linked_files[key]['image_filepath']
        image_filename = linked_files[key]['image_filename']
        annotation_filename = linked_files[key]['annotation_filename']

        if split_decision(args.train_percentage):
            new_anno_path = os.path.join(train_dirs['annotations'], annotation_filename)
            new_image_path = os.path.join(train_dirs['images'], image_filename)
        else:
            new_anno_path = os.path.join(val_dirs['annotations'], annotation_filename)
            new_image_path = os.path.join(val_dirs['images'], image_filename)

        move_files(current_anno_path, current_image_path, new_anno_path, new_image_path)
        i += 1





if __name__ == '__main__':
    main()