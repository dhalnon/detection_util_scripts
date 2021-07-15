import os
import argparse
import numpy as np
import cv2
import xmltodict
import time
import multiprocessing as mp
from pascal_voc_writer import Writer
import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
       description='Use to resize images for tensorflow',
       formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-a', 
                        '--annotations',
                        default='/home/ftdavid/training_data/face_detector_retrain/ICI_outdoor_faces_in_cars/annotations',
                        type=str, 
                        help='Path to the annotations directory')
    parser.add_argument('-i',
                        '--images',
                        default='/home/ftdavid/training_data/face_detector_retrain/ICI_outdoor_faces_in_cars/images',
                        type=str,
                        help='Path to the image directory')
    parser.add_argument('-s', 
                        '--image_size',
                        default=320,
                        type=int, 
                        help='Final dimension to resize images to, images will be made square')

    return parser.parse_args()


def load_file_list(annotations_directory, images_directory):
    anno_list = [os.path.join(root, f) for root,_,files in os.walk(annotations_directory) for f in files if f.endswith('.xml')]
    anno_dict = {f.split(os.sep)[-1].replace('.xml',''):f for f in anno_list}
    im_list = [os.path.join(root, f) for root,_,files in os.walk(images_directory) for f in files if f.endswith('.jpg')]
    im_dict = {f.split(os.sep)[-1]:f for f in im_list}
    return anno_dict, im_dict


def link_anno_to_image(annotation_file, image_dict, resize_to):
    return {'annotation_file':list(annotation_file.values())[0], 'image_file':image_dict[list(annotation_file.keys())[0]], 'smol_size':resize_to}


def get_object(objects):
    if isinstance(objects, list):
        return objects[0]
    return objects


def resize_linked_files(linked_files):
    def shrink_image(example, smol_size):
        smol_image_filename = os.path.join(example['image_filepath'].replace('images',f'images_resized_{smol_size}x{smol_size}'))
        smol_anno_filename = os.path.join(example['annotation_filepath'].replace('annotations',f'annotations_resized_{smol_size}x{smol_size}'))
        for filename in [smol_anno_filename, smol_image_filename]:
            os.makedirs(os.path.join(os.sep, *(filename.split(os.sep)[:-1])), exist_ok=True)
        writer = Writer(smol_image_filename, smol_size, smol_size)
        annotation = example['annotation']['annotation']
        (W, H) = int(annotation['size']['width']), int(annotation['size']['height'])
        W_transform, H_transform = smol_size / W, smol_size / H
        objects = [get_object(annotation[x]) for x in annotation.keys() if x == 'object']
        for obj in objects:
            new_obj = obj
            bbox = obj['bndbox']
            for coord in bbox.keys():
                if coord.startswith('x'):
                    mult = W_transform
                elif coord.startswith('y'):
                    mult = H_transform
                new_obj['bndbox'][coord] = int(int(bbox[coord]) * mult)
            writer.addObject(obj['name'],*new_obj['bndbox'].values())
        writer.save(smol_anno_filename)
        cv2.imwrite(smol_image_filename, cv2.resize(image, [smol_size,smol_size]))

    with open(linked_files['annotation_file']) as anno:
        annotation = xmltodict.parse(anno.read())
    image = cv2.imread(linked_files['image_file'])
    example =  {'annotation':annotation,'image':image,'image_filepath':linked_files['image_file'],'annotation_filepath':linked_files['annotation_file']}
    shrink_image(example, linked_files['smol_size'])


def main():

    pool = mp.Pool(mp.cpu_count())
    
    startT = time.time_ns()
    args = parse_args()
    anno_dict, im_dict = load_file_list(args.annotations, args.images)

    linked_images = [link_anno_to_image({anno_key:anno_dict[anno_key]}, im_dict, args.image_size) for anno_key in anno_dict.keys()]
    for _ in tqdm.tqdm(pool.imap_unordered(resize_linked_files, [*linked_images], 100), total=len(linked_images)):
        pass

    print((time.time_ns() - startT)/10**9)



if __name__ == '__main__':
    main()
    
