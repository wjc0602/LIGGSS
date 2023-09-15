"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import ipdb
from data.base_dataset import BaseDataset, get_params, get_transform
from PIL import Image
import util.util as util
import os


class Pix2pixDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--no_pairing_check', action='store_true',
                            help='If specified, skip sanity check of correct label-image file pairing')
        return parser


    def initialize(self, opt):
        self.opt = opt

        label_paths, image_paths = self.get_paths(opt)
        util.natural_sort(label_paths)
        util.natural_sort(image_paths)

        label_paths = label_paths[:opt.max_dataset_size]
        image_paths = image_paths[:opt.max_dataset_size]

        # if not opt.no_pairing_check:
        #     for path1, path2 in zip(label_paths, image_paths):
        #         assert self.paths_match(path1, path2), \
        #             "The label-image pair (%s, %s) do not look like the right pair because the filenames are quite different. Are you sure about the pairing? Please see data/pix2pix_dataset.py to see what is going on, and use --no_pairing_check to bypass this." % (
        #                 path1, path2)

        self.label_paths = label_paths
        self.image_paths = image_paths

        size = len(self.label_paths)
        self.dataset_size = size

    def get_paths(self, opt):
        label_paths = []
        image_paths = []
        instance_paths = []
        assert False, "A subclass of Pix2pixDataset must override self.get_paths(self, opt)"
        return label_paths, image_paths, instance_paths

    def paths_match(self, path1, path2):
        filename1_without_ext = os.path.splitext(os.path.basename(path1))[0]
        filename2_without_ext = os.path.splitext(os.path.basename(path2))[0]
        return filename1_without_ext == filename2_without_ext

    def __getitem__(self, index):
        # Label Image
        label_path = self.label_paths[index]
        label = Image.open(label_path)
        params = get_params(self.opt, label.size)

        transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        transform_image = get_transform(self.opt, params)

        label_tensor = transform_label(label)

        # input image (real images)
        if len(self.image_paths) == 0:
            image_tensor = 0
            image_path = label_path
        else:
            if self.name == "CustomDataset" and not self.isTrain:
                label_file = label_path.split("/")[-1]
                img_file = self.ref_dict[label_file].replace(".png", ".jpg")
                image_path = "/".join([self.image_dir, img_file])
                ref_image_name = self.ref_dict[label_file].replace(".png", ".jpg")
                ref_image_path = "/".join([self.image_dir, ref_image_name])
            else:
                image_path = self.image_paths[index]
                image_name = image_path.split("/")[-1]
                ref_image_name = self.ref_dict[image_name]
                ref_image_path = "/".join([self.image_dir, ref_image_name])

                assert self.paths_match(label_path, image_path), \
                    "The label_path %s and image_path %s don't match." % \
                    (label_path, image_path)

            image = Image.open(image_path)
            image = image.convert('RGB')
            image_tensor = transform_image(image)

        # ref_label
        ref_label_path = ref_image_path.replace('imgs', 'labels').replace('.jpg', '.png')
        ref_label = Image.open(ref_label_path)
        ref_label_tensor = transform_label(ref_label)
        # ref_img
        ref_image_tensor = transform_image(Image.open(ref_image_path).convert('RGB'))

        assert self.paths_match(ref_label_path, ref_image_path), \
            "The ref_label_path %s and ref_image_path %s don't match." % \
            (label_path, image_path)

        input_dict = {'label': label_tensor,
                      'image': image_tensor,
                      'path': image_path if self.isTrain else label_path,
                      'ref_label': ref_label_tensor,
                      'ref_image': ref_image_tensor,
                      }

        # Give subclasses a chance to modify the final output
        self.postprocess(input_dict)
        # ipdb.set_trace()
        return input_dict

    def postprocess(self, input_dict):
        return input_dict

    def __len__(self):
        return self.dataset_size
