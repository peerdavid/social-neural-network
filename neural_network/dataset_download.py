#
# Download images of a single class (cars, books etc.) from flickr
# using datr - http://github.com/peerdavid/datr/
#

import datr
import sys

def main(argv):
    path = argv[0]
    name = argv[1]
    max_num_images = int(argv[2])

    datr.download(
        path=path, 
        search_tags=name, 
        use_free_text=False, 
        license="", 
        max_num_img=max_num_images, 
        num_threads=20, 
        image_size="m")


if __name__ == "__main__":
   main(sys.argv[1:])

