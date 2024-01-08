import open3d as o3d
import numpy as np
import os
import sys
from pc_util import extract_knn_patch, normalize_point_cloud, farthest_point_sample, group_points,group_points_with_normal


raw_data_path='/home/gpuadmin/IJS/Color_PUGeoNet/data/'
save_root_path='/home/gpuadmin/IJS/Color_PUGeoNet/data/test_patches/'

basic_num=10000
num_point_per_patch=64
num_patch=int(basic_num/num_point_per_patch*3)


def build_dataset(up_ratio_list,mode):
    for up_ratio in up_ratio_list:
        try:
            os.mkdir(os.path.join(save_root_path,'%d'%up_ratio))

        except:
            pass
    try:
        os.mkdir(os.path.join(save_root_path, 'basic'))
    except:
        pass


    
    input_data_path_list =os.listdir(os.path.join(raw_data_path, 'train_%d' % (int(basic_num))))
    i=0
    for input_data_name in input_data_path_list:
        pcd = o3d.io.read_point_cloud(os.path.join(os.path.join(raw_data_path, '%s_%d' % (mode, int(basic_num)),input_data_name)))
        points = np.asarray(pcd.points)

        rgb = np.array(pcd.colors).astype(np.float32) 
        normal = np.array(pcd.normals).astype(np.float32)
        input_data = np.concatenate((points, normal), axis=1)
        input_data = np.concatenate((input_data, rgb), axis=1)

        xyz,centroid,furthest_distance=normalize_point_cloud(input_data[:,0:3])
        input_data[:,0:3]=xyz
        centroid_points=farthest_point_sample(input_data,num_patch)
        input_patches=group_points_with_normal(input_data,centroid_points,num_point_per_patch)
        print(input_patches.shape)
        normalized_input_patches_xyz,centroid_patches,furthest_distance_patches=normalize_point_cloud(input_patches[:,:,0:3])
        input_patches[:,:,0:3]=normalized_input_patches_xyz
        np.save(os.path.join(save_root_path,'basic','%d.npy'%i),input_patches)
        for up_ratio in up_ratio_list:
            label_data = o3d.io.read_point_cloud(os.path.join(os.path.join(raw_data_path, '%s_%d' % (mode, int(basic_num*up_ratio)),input_data_name)))
            label_data_points = np.asarray(label_data.points)
            label_data_rgb = np.asarray(label_data.colors).astype(np.float32)
            label_data_normal = np.asarray(label_data.normals).astype(np.float32)
            label_input_data = np.concatenate((label_data_points, label_data_normal), axis=1)
            label_input_data = np.concatenate((label_input_data, label_data_rgb), axis=1)

            xyz=(label_input_data[:,0:3]-centroid)/furthest_distance
            label_input_data[:,0:3]=xyz
            label_patches=group_points_with_normal(label_input_data,centroid_points,num_point_per_patch*up_ratio)
            label_patches_xyz=(label_patches[:,:,0:3]-centroid_patches)/furthest_distance_patches
            label_patches[:,:,0:3]=label_patches_xyz
            np.save(os.path.join(save_root_path,'%d'%up_ratio,'%d.npy'%i),label_patches)

        i=i+1

if __name__=='__main__':
    up_ratio_list = [4]
    build_dataset(up_ratio_list,'train')