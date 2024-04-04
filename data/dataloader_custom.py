import glob
import numpy as np
from PIL import Image

import torch
import torch.utils.data.distributed
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

####################### added changes from geonet #########################
# import tensorflow as tf
import scipy.misc
import scipy.io
# import cv2

####################### added changes from geonet #########################
class CustomLoader(object):
    def __init__(self, args, fldr_path):
        self.testing_samples = CustomLoadPreprocess(args, fldr_path)
        self.data = DataLoader(self.testing_samples, 1,
                               shuffle=False,
                               num_workers=1,
                               pin_memory=False)


class CustomLoadPreprocess(Dataset):
    def __init__(self, args, fldr_path):
        self.fldr_path = fldr_path
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.filenames = glob.glob(self.fldr_path + '/*.png') + glob.glob(self.fldr_path + '/*.jpg')
        self.input_height = args.input_height
        self.input_width = args.input_width

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        img = Image.open(img_path).convert("RGB").resize(size=(self.input_width, self.input_height), resample=Image.BILINEAR)
        img = np.array(img).astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)
        img = self.normalize(img)

        img_name = img_path.split('/')[-1]
        img_name = img_name.split('.png')[0] if '.png' in img_name else img_name.split('.jpg')[0]

        sample = {'img': img,
                  'img_name': img_name}

        return sample

####################### added changes from geonet #########################
# class MatDataLoader(Dataset):

#     def __init__(self,file_path):

#         """
#         file_path : complete file path for the matlab file.
#         """
        
#         self.matdata_path = file_path
        
#         with open(self.matdata_path , 'r') as f:
#             self.filenames = f.readlines()

        
#         print("\n\nLength of file names : ",len(self.filenames))

#     # data process
#     def myfunc(self,x):
#         try:
#             data_dic = scipy.io.loadmat(x)
#             data_img = data_dic['img']
#             #print "aaaaa"
#             data_depth = data_dic['depth']
#             depth_mask = np.zeros((crop_size_h, crop_size_w), dtype=np.float32)
#             depth_mask[np.where(data_depth < 0.1)] = 0.0
#             depth_mask[np.where(data_depth >= 0.1)] = 1.0
#             data_norm = data_dic['norm']
#             data_mask = data_dic['mask']
#             grid = data_dic['grid']
#         except:
#             data_img = np.zeros((crop_size_h, crop_size_w, 3), dtype=np.float32)
#             data_depth = np.zeros((crop_size_h, crop_size_w), dtype=np.float32)
#             data_mask = np.zeros((crop_size_h, crop_size_w), dtype=np.float32)
#             data_norm = np.zeros((crop_size_h, crop_size_w, 3), dtype=np.float32)
#             depth_mask = np.zeros((crop_size_h, crop_size_w), dtype=np.float32)
#             grid = np.zeros((crop_size_h, crop_size_w,3), dtype=np.float32)

#         return data_img, data_depth, data_norm, data_mask,depth_mask,grid

#     def input_producer(self):
#         def read_data():
#             image, depth, norm,mask, depth_mask, grid= tf.py_func(self.myfunc,[self.data_queue[0]],[tf.float32,tf.float32,tf.float32,tf.float32,tf.float32,tf.float32])
#             image, depth, norm, mask, depth_mask, grid = preprocessing(image, depth, norm, mask,depth_mask,grid)
#             return image, depth, norm, mask, depth_mask,grid

#         # data loader + data augmentation
#         def preprocessing(image, depth, norm, mask,depth_mask, grid):

#             image = tf.cast(image, tf.float32)
#             depth = tf.cast(depth, tf.float32)
#             norm = tf.cast(norm, tf.float32)
#             mask = tf.cast(mask, tf.float32)
#             depth_mask = tf.cast(depth_mask, tf.float32)
#             grid = tf.cast(grid, tf.float32)
#             random_num = tf.random_uniform([], minval=0, maxval=1.0, dtype=tf.float32, seed=None, name=None)

#             mirror_cond = tf.less(random_num, 0.5)
#             stride = tf.where(mirror_cond, -1, 1)
#             image = image[:, ::stride, :]
#             depth = depth[:, ::stride]
#             mask = mask[:, ::stride]
#             depth_mask = depth_mask[:, ::stride]
#             norm = norm[:, ::stride, :]
#             norm_x, norm_y, norm_z = tf.split(value=norm, num_or_size_splits=3, axis=2)
#             norm_x = tf.scalar_mul(tf.cast(stride, dtype=tf.float32), norm_x)
#             norm = tf.cast(tf.concat([norm_x, norm_y, norm_z], 2), dtype=tf.float32)


#             img_r, img_g, img_b = tf.split(value=image, num_or_size_splits=3, axis=2)
#             image = tf.cast(tf.concat([img_b, img_g, img_r], 2), dtype=tf.float32)

#             image.set_shape((crop_size_h, crop_size_w, 3))
#             depth.set_shape((crop_size_h, crop_size_w))
#             norm.set_shape((crop_size_h, crop_size_w, 3))
#             mask.set_shape((crop_size_h, crop_size_w))
#             depth_mask.set_shape((crop_size_h, crop_size_w))
#             grid.set_shape((crop_size_h,crop_size_w,3))
#             return image, depth, norm, mask, depth_mask,grid

#         with tf.variable_scope('input'):
#             imglist = tf.convert_to_tensor(self.data_list, dtype=tf.string)
#             self.data_queue = tf.train.slice_input_producer([imglist], capacity=100)
#             images, depths,norms,masks,depth_masks, grid = read_data()
#             batch_images, batch_depths, batch_norms, batch_masks,batch_depth_masks, grid = tf.train.batch([images, depths, norms, masks,depth_masks, grid], batch_size=self.batch_size, num_threads=4, capacity=60)


#         return batch_images, batch_depths, batch_norms, batch_masks,batch_depth_masks, grid

#     def __len__(self):
#         return len(self.filenames)


#     def __getitem__(self, idx):
#         mat_file_path = self.filenames[idx]

#         batch_images, batch_depths, batch_norms, batch_masks,batch_depth_masks, grid = self.input_producer()

#         data_dictionary = {"img":batch_images,
#                         "depth":batch_depths, 
#                         "norm":batch_norms,
#                         "norm_valid_mask":batch_masks,
#                         "depth_masks":batch_depth_masks,
#                         "grid":grid
#                         }

#         return data_dictionary  



# ####################### added changes from geonet #########################