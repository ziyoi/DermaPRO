#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from isicdata import download_archive
from tqdm import tqdm
import sys
import random
import io
import contextlib2
import tensorflow as tf
import PIL
import glob
import cv2
import pprint
from skimage import measure

def diagnosis_from_description(description):
    """ Return the diagnosis in each description """
    diagnosis = description["meta"]["clinical"]["diagnosis"]
    if diagnosis not in ["nevus", "melanoma", "seborrheic keratosis"]:
        raise ValueError(diagnosis)
    return diagnosis


def download(num_images, offset, num_cpus):
    img_ids = download_archive.get_images_ids(num_images, offset)

    num_images_found = len(img_ids)
    if num_images is None or num_images_found == num_images:
        print('Found {0} images'.format(num_images_found))
    else:
        num_images = num_images_found
        print('Found {0} images and not the requested {1}'.format(
            num_images_found, num_images))

    descriptions = download_archive.download_descriptions(
        img_ids, "/tensorflow/data/desc/", num_cpus)
    #print("\nDescription of first image:\n", json.dumps(descriptions[0], indent=4, sort_keys=True))
    #diagnosis = diagnosis_from_description(descriptions[0])
    #print("\nDiagnosis of first image: {0}\n".format(diagnosis))

    #download_archive.download_images(descriptions, "/tensorflow/data/img/", num_cpus)
    #download_archive.download_segmentations(
    #    descriptions, "/tensorflow/data/seg/", None, num_cpus)
    return descriptions


def image_path_from_desc(desc):
    """ Get the path to the image from its description. """
    return "/tensorflow/data/img/" + desc["name"] + ".jpeg"


def seg_path_from_desc(desc):
    """ Get the path to the segmentation image from its description """
    paths = glob.glob("/tensorflow/data/seg/" + desc["name"] + "*.png")
    assert len(paths) == 1
    return paths[0]


def seg_bbox(desc) -> str:
    """ Get the bounding box of the segmentation based on the description. """
    path = seg_path_from_desc(desc)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    lbl = measure.label(img)
    props = measure.regionprops(lbl)
    assert len(props) == 1
    return props[0].bbox


def plot_img_with_bbox(desc):
    """ Plot the image with the bounding box overlayed. """
    ax = plt.axes()
    img = cv2.imread(image_path_from_desc(desc))
    bbox = seg_bbox(desc)
    cv2.rectangle(img, (bbox[1], bbox[0]), (bbox[3], bbox[2]), (255, 0, 0), 2)
    ax.imshow(img)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(desc["name"])
    plt.show()


def label_map_dict(descriptions):
    """ Create a label map dictionary from the descriptions. """
    i = 1
    lmd = {}
    for d in descriptions:
        try:
            diag = diagnosis_from_description(d)
        except (KeyError, ValueError):
            continue
        if diag and (diag not in lmd):
            lmd[diag] = i
            i = i + 1
    return lmd


def write_label_map_dict(lmd, filename):
    """ Write the label map dict to the required format specified in
    https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md.
    """
    end = '\n'
    s = ' '
    out = ''
    for name in lmd:
        out += 'item' + s + '{' + end
        out += s * 2 + 'id:' + ' ' + str(lmd[name]) + end
        out += s * 2 + 'name:' + ' ' + '\'' + name + '\'' + end
        out += '}' + end * 2
    with open(filename, 'w') as f:
        f.write(out)


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def description_to_tf_example(description,
                              label_map_dict):
    """Create tf.Example proto based on description.
    Adapted from https://github.com/tensorflow/models/blob/master/research/object_detection/dataset_tools/create_pet_tf_record.py
    Notice that this function normalizes the bounding box coordinates provided
    by the raw data.
    Args:
      description: Image description
      label_map_dict: A map from string label names to integers ids.
    Returns:
      example: The converted tf.Example.
    Raises:
      ValueError: if the image pointed to by data['filename'] is not a valid JPEG
    """

    img_path = image_path_from_desc(description)
    with tf.gfile.GFile(img_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')

    width = int(image.size[0])
    height = int(image.size[1])

    bbox = seg_bbox(description)
    xmins = [float(bbox[1]) / width]
    ymins = [float(bbox[0]) / height]
    xmaxs = [float(bbox[3]) / width]
    ymaxs = [float(bbox[2]) / height]

    diag = diagnosis_from_description(description)
    assert diag
    classes = [label_map_dict[diag]]
    classes_text = [diag.encode('utf8')]

    feature_dict = {
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
        'image/filename': bytes_feature(
            description["name"].encode('utf8')),
        'image/source_id': bytes_feature(
            description["name"].encode('utf8')),
        'image/encoded': bytes_feature(encoded_jpg),
        'image/format': bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': float_list_feature(xmins),
        'image/object/bbox/xmax': float_list_feature(xmaxs),
        'image/object/bbox/ymin': float_list_feature(ymins),
        'image/object/bbox/ymax': float_list_feature(ymaxs),
        'image/object/class/text': bytes_list_feature(classes_text),
        'image/object/class/label': int64_list_feature(classes),
    }

    example = tf.train.Example(
        features=tf.train.Features(feature=feature_dict))
    return example


def open_sharded_output_tfrecords(exit_stack, base_path, num_shards):
    """Opens all TFRecord shards for writing and adds them to an exit stack.
    From https://github.com/tensorflow/models/blob/master/research/object_detection/dataset_tools/create_pet_tf_record.
    Args:
      exit_stack: A context2.ExitStack used to automatically closed the TFRecords
        opened in this function.
      base_path: The base path for all shards
      num_shards: The number of shards
    Returns:
      The list of opened TFRecords. Position k in the list corresponds to shard k.
    """
    tf_record_output_filenames = [
        '{}-{:05d}-of-{:05d}'.format(base_path, idx, num_shards)
        for idx in range(num_shards)
    ]

    tfrecords = [
        exit_stack.enter_context(tf.python_io.TFRecordWriter(file_name))
        for file_name in tf_record_output_filenames
    ]

    return tfrecords


def write_tfrecords(descriptions, num_shards, output_filename, label_map_dict):
    """ Create a tfrecords file from the given descriptions """
    with contextlib2.ExitStack() as tf_record_close_stack:
        output_tfrecords = open_sharded_output_tfrecords(
            tf_record_close_stack, output_filename, num_shards)
        for idx in tqdm(range(len(descriptions))):
            description = descriptions[idx]

            try:
                tf_example = description_to_tf_example(
                    description,
                    label_map_dict)
                if tf_example:
                    shard_idx = idx % num_shards
                    output_tfrecords[shard_idx].write(
                        tf_example.SerializeToString())
            except (ValueError, AssertionError, KeyError, tf.errors.NotFoundError):
                tqdm.write('Invalid example: %s, ignoring.' %
                           description["name"])

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_images', type=int,
                        help='The number of images you would like to download from the ISIC Archive. '
                             'Leave empty to download all the available images', default=None)
    parser.add_argument('--offset', type=int, help='The offset of the image index from which to start downloading',
                        default=0)
    parser.add_argument('--p', type=int, help='The number of processes to use in parallel', default=16)
    parser.add_argument('--training_frac', type=float, help='The fraction of data for use in training', default=0.9)
    parsed_args = parser.parse_args(args)
    return parsed_args

def main(args):
    args = parse_args(args)
    descriptions = download(args.num_images, args.offset, args.p)

    label_map_file = "/tensorflow/data/label_map_dict.pbtxt"
    lmd = label_map_dict(descriptions)
    write_label_map_dict(lmd, label_map_file)

    shuffled_descriptions = descriptions.copy()
    random.seed(42)
    random.shuffle(shuffled_descriptions)
    num_examples = len(shuffled_descriptions)
    num_train = int(args.training_frac * num_examples)
    train_examples = shuffled_descriptions[:num_train]
    val_examples = shuffled_descriptions[num_train:]
    print('%d training and %d validation examples.'%
                   (len(train_examples), len(val_examples)))

    train_file = "/tensorflow/data/training_data.tfrecords"
    eval_file = "/tensorflow/data/validation_data.tfrecords"
    write_tfrecords(train_examples, 10, train_file, lmd)
    write_tfrecords(val_examples, 10, eval_file, lmd)

if __name__ == "__main__":
    main(sys.argv[1:])
