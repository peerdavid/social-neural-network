################################################
# Social Neural Network
# Peer David - 2017
################################################

################################################
# Note: We preprocess the images on disk, so the
# computation has to be done only once, and not
# multiple times during training...
################################################

import sys
from PIL import Image
import tensorflow as tf
import subprocess

import params
import data_input

FLAGS = params.FLAGS


def crop_image_bounding_box(box, image, image_path):
    expand = FLAGS.expand_bounding_box
    box_x1 = max(0, box[0]-expand)
    box_y1 = max(0, box[1]-expand)
    box_x2 = min(image.size[0], box[2]+expand)
    box_y2 = min(image.size[1], box[3]+expand)

    return image.crop((box_x1, box_y1, box_x2, box_y2))


def resize_image_keep_ratio(image_path, size):
    proc = subprocess.Popen("mogrify -resize {0}x{1} {2}"
        .format(size[0], size[1], image_path), shell=True)
    proc.wait()

def resize_image_fixed_size(image_path, size):
    proc = subprocess.Popen("mogrify -extent {0}x{1} -gravity Center -background black -colorspace RGB {2}"
        .format(size[0], size[1], image_path), shell=True)
    proc.wait()


def resize_smaller_side(image_path, image, size):
    resize_cmd = "{0}x".format(size[0]) if image.size[0] < image.size[1] else "x{0}".format(size[1])
    proc = subprocess.Popen("mogrify -resize {0} {1}"
        .format(resize_cmd, image_path), shell=True)
    proc.wait()

#
# M A I N
#   
def main(argv=None):
    filenames, labels, num_classes = data_input.read_labeled_image_list(FLAGS.img_dir)

    print("Preprocessing %s" % FLAGS.img_dir)

    num_images = len(filenames)
    for i in range(num_images):
        sys.stdout.write("  Preprocessing images...%d%%\r" % (i * 100 / num_images))
        sys.stdout.flush()

        image_path = filenames[i]
        size = (FLAGS.orig_image_width, FLAGS.orig_image_height)
        resize_image_keep_ratio(image_path, size)
        resize_image_fixed_size(image_path, size)
    print("Done.")
    
            

if __name__ == '__main__':
    tf.app.run()