import os
import os.path as osp
import numpy as np
import cv2
import matplotlib.pyplot as plt


def main():
    """Main for the program for getting image stats."""
    base_dir = '/home/sjimenez/imagenes_prueba'
    out_dir = '/home/sjimenez/imagenes_prueba'
    for _, _, files in os.walk(base_dir, topdown=False):
        for f in files:
            print('--------- {} ---------'.format(f))
            act_dir = osp.join(base_dir, f)
            act_im = cv2.imread(act_dir)
            if act_im is not None:
                get_image_stats(act_im, out_dir, f)


def get_image_stats(image, out_dir, cur_file):
    """Retrieve important information of the image."""
    # Output directory
    output_base = osp.join(out_dir, cur_file.split('.')[0])
    os.mkdir(output_base)
    # Print dimensions of the image
    width, height, color = image.shape
    print('The resolution of the image if of {}x{}x{}'.format(width,
                                                              height,
                                                              color))
    print('Total of {} pixels'.format(width * height * color))

    # Get histogram
    print('Calculating histogram')
    flat_img = image.mean(axis=2).flatten()
    counts, bins = np.histogram(flat_img, range(257))
    plt.bar(bins[:-1], counts, width=1, edgecolor='none')
    output_file = osp.join(out_dir, output_base, 'histogram.png')
    plt.xlabel('Intensidad')
    plt.ylabel('Número de pixeles')
    print('Saving histogram')
    plt.savefig(output_file, bbox_inches='tight')

    # LAB space
    lab_image = cv2.cvtColor(image[8000:8500, 8000:8500, :], cv2.COLOR_BGR2LAB)
    output_file = osp.join(out_dir, output_base, 'lab.png')
    cv2.imwrite(output_file, lab_image)


main()
