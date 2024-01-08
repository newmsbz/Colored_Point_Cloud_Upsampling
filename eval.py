import sys
import os
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
import torch
import numpy as np
import argparse
import model
import open3d as o3d
from glob import glob
from pc_util import  normalize_point_cloud, farthest_point_sample, group_points_with_normal
from skimage.color import hsv2rgb

def eval_patches(input_data, arg, model):
    centroids = farthest_point_sample(input_data[:,0:3], arg.num_patch)
    # print(input_data.shape)
    # patches = group_points(xyz, centroids, arg.num_point)
    patches = group_points_with_normal(input_data, centroids, arg.num_point)
    # print(patches.shape)
    normalized_patches, patch_centroid, furthest_distance = normalize_point_cloud(patches[:,:,0:3])
    patches[:,:,0:3]=normalized_patches

    dense_patches_list = []
    dense_normal_list = []
    sparse_normal_list = []
    rgb_list = []

    #print(normalized_patches.shape)

    # for i in range(normalized_patches.shape[0]):
    for i in range(normalized_patches.shape[0]):
        # print(normalized_patches.shape)
        input_data2=torch.from_numpy(patches[i]).unsqueeze(0).transpose(1, 2).cuda()
        #print(torch.from_numpy(normalized_patches[i]).size())
        dense_patches, dense_normal, sparse_normal, rgb = model(input_data2)
        dense_patches_list.append(dense_patches.transpose(1, 2).detach().cpu().numpy())
        dense_normal_list.append(dense_normal.transpose(1, 2).detach().cpu().numpy())
        sparse_normal_list.append(sparse_normal.transpose(1, 2).detach().cpu().numpy())
        # print(rgb)
        rgb_list.append(rgb.transpose(1, 2).detach().cpu().numpy())

    # print(dense_patches_list[0].shape, dense_normal_list[0].shape, sparse_normal_list[0].shape, rgb_list[0].shape)
    gen_ddense_xyz = np.concatenate(dense_patches_list, axis=0)
    # print(gen_ddense_xyz.shape, furthest_distance.shape, patch_centroid.shape)
    gen_ddense_xyz = (gen_ddense_xyz * furthest_distance) + patch_centroid
    gen_ddense_normal = np.concatenate(dense_normal_list, axis=0)

    return np.reshape(gen_ddense_xyz, (-1, 3)), np.reshape(gen_ddense_normal, (-1, 3)), np.reshape(rgb_list, (-1, 3))


def evaluate(model, arg):
    model.eval()
    shapes = glob(arg.eval_ply + '/*.ply')    # original
    # shapes = glob(arg.eval_ply + '/*.xyz')

    for i, item in enumerate(shapes):
        #print(item)
        obj_name = item.split('/')[-1]

        # data = np.loadtxt(item)
        # input_sparse_xyz = data[:, 0:3]
        # input_sparse_normal = data[:, 3:6]
        # normalize_sparse_xyz, centroid, furthest_distance = normalize_point_cloud(input_sparse_xyz)

        pcd = o3d.io.read_point_cloud(item)
        points = np.asarray(pcd.points)
        rgb = np.array(pcd.colors).astype(np.float32) #(N,3 : rgbn colors)
        # rgb = np.zeros_like(pcd.points).astype(np.float32)
        # normals = np.array(pcd.normals).astype(np.float32)
        input_data = np.concatenate((points, rgb), axis=1)
        # input_data = np.concatenate((input_data, rgb), axis=1)
        # normalize_sparse_xyz, centroid, furthest_distance = normalize_point_cloud(input_data[:,0:3])
        normalize_sparse_xyz, centroid, furthest_distance = normalize_point_cloud(points)
        input_data[:,0:3] = normalize_sparse_xyz

        
        dense_xyz, dense_normal, color = eval_patches(input_data, arg, model)
        dense_xyz = dense_xyz * furthest_distance + centroid
        gen_dense=np.concatenate((dense_xyz,dense_normal),axis=-1)
        gen_dense=np.concatenate((gen_dense, color), axis=-1)
        # print(gen_dense.shape, dense_xyz.shape, dense_normal.shape, color.shape)
        #print(arg.eval_save_path)
        savepath=os.path.join(arg.eval_save_path,obj_name)
        #print(arg.eval_save_path)
        #print(savepath)
        
        #print(gen_dense.shape)
        gen_dense=farthest_point_sample(gen_dense,arg.num_shape_point*arg.up_ratio)

        # print(gen_dense)
        # print(gen_dense.shape)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(gen_dense[:,0:3])
        pcd.normals = o3d.utility.Vector3dVector(gen_dense[:,3:6])
        pcd.colors = o3d.utility.Vector3dVector(gen_dense[:,6:9])   # rgb 저장

        # 
        # hsv = gen_dense[:,6:9]
        # rgb = hsv2rgb(hsv)
        # print(hsv)
        # print(rgb)
        # pcd.colors = o3d.utility.Vector3dVector(rgb)
        # pcd.colors = o3d.utility.Vector3dVector(gen_dense[:,6:9].astype(np.float) / 255.0)
        o3d.io.write_point_cloud(savepath, pcd)
        # print(color.shape)
        # pcd.rgb = 


        #print(gen_dense.shape)
        # np.savetxt(savepath,gen_dense)
        print(obj_name,'is saved')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--up_ratio', type=int, default=4, choices=[4,8,12,16], help='Upsampling Ratio')  
    parser.add_argument('--model', default='model_pugeo', help='Model for upsampling')
    parser.add_argument('--num_point', type=int, default=64,choices=[64], help='Point Number')
    
    parser.add_argument('--eval_ply', default='/home/gpuadmin/IJS/Color_PUGeoNet/other_num/test_10000', help='Folder to store point cloud(ply format) toevaluate')
    # parser.add_argument('--eval_ply', default='/data/IJS/IJS/dataset/pugeo_original_dataset/mesh/test_5000', help='Folder to store point cloud(ply format) toevaluate')
    # parser.add_argument('--eval_ply', default='/home/gpuadmin/IJS/Color_PUGeoNet/new_dataset/downsample_10000/', help='Folder to store point cloud(ply format) toevaluate')
    parser.add_argument('--num_shape_point', type=int, default=10000, help='Point Number per shape')
    parser.add_argument('--patch_num_ratio', type=int, default=3, help='Number of points covered by patch')
    arg = parser.parse_args()
    # arg.log_dir=os.path.join(os.path.dirname(__file__),'..','training_result/original(color_mesh_data_log_x%d)'%arg.up_ratio)
    arg.log_dir=os.path.join(os.path.dirname(__file__),'..','new_train_result/new_ver27_log_x%d'%arg.up_ratio)
    
    arg.num_patch = int(arg.num_shape_point / arg.num_point * arg.patch_num_ratio)
    # arg.eval_save_path=os.path.join(os.path.dirname(__file__),'..','eval_result/original/x%d/grey_test_1000'%arg.up_ratio)
    arg.eval_save_path=os.path.join(os.path.dirname(__file__),'..','new_ver_save/ver27/x%d/'%arg.up_ratio)
    try:
        os.mkdir(arg.eval_save_path)
    except:
        pass
    model = model.pugeonet(up_ratio=arg.up_ratio, knn=15)
    model = model.cuda()

    model.load_state_dict(torch.load(os.path.join(arg.log_dir,'model.t7')))
    evaluate(model, arg)
