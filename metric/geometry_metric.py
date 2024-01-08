import open3d as o3d
import os
import numpy as np
import point_cloud_utils as pcu
from pc_util import normalize_point_cloud
from scipy.stats import wasserstein_distance
from scipy.spatial import distance
import warnings
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import pymesh


def cal_cd(gt, result):
    chamfer_dist = pcu.chamfer_distance(gt, result)
    return chamfer_dist


def cal_hd(gt, result):
    # Take a max of the one sided squared  distances to get the two sided Hausdorff distance
    hausdorff_dist = pcu.hausdorff_distance(gt, result)
    return hausdorff_dist


def cal_emd(a,b):
    a=np.reshape(a, a.size)
    b=np.reshape(b, b.size)
    EMD_dist=wasserstein_distance(a,b)
    return EMD_dist


def cal_jsd(a,b):
    a=np.reshape(a, a.size)
    b=np.reshape(b, b.size)
    JSD_dist=distance.jensenshannon(a,b)

    return JSD_dist

gt_path = './test_data/x4/gt_xyz/'
# gt_path = './test_data/x4/gt_off/'
# gt_path = '/data/IJS/IJS/dataset/pugeo_original_dataset/mesh/test_20000/'    # sketchFab gt data(20000 points)

result_path = './test_data/x4/ours_xyz/'    # ours
# result_path = './test_data/ver25_600/xyz/'
# result_path = './test_data/x4/colorup/'    # colorup
# result_path = './test_data/x4/pugeonet/'    # pugeonet
# result_path = '/home/gpuadmin/IJS/PU-Net_pytorch/sketchfab_test_output/'  # punet
# result_path = '/home/gpuadmin/IJS/Color_PUGeoNet/new_ver_save/ver25_epoch_600/x4/' # ours(SketchFab test data, 5000 points)
# result_path = '/home/gpuadmin/IJS/Color_PUGeoNet/test_data/x4/ours_xyz/' # ours(SketchFab test data, 
# result_path = '/home/gpuadmin/IJS/PUGeoNet_pytorch/pretrained_weight_x4_sketchfab_test_data/'

name_list = os.listdir(gt_path)
cd_list = []
hd_list = []
emd_list = []
jsd_list = []
p2f_list = []


for name in name_list:
    print(name)
    gt_pcd_name = gt_path + name
    result_pcd_name = result_path + name
    # result_pcd_name = result_path + name[:-3] + 'xyz'
    
    gt_pcd = o3d.io.read_point_cloud(gt_pcd_name)
    result_pcd = o3d.io.read_point_cloud(result_pcd_name)

    gt_xyz = np.array(gt_pcd.points).astype(np.float32)
    gt_xyz_norm, _ , _ = normalize_point_cloud(gt_xyz)
    result_xyz = np.array(result_pcd.points).astype(np.float32)
    result_xyz_norm, _ , _ = normalize_point_cloud(result_xyz)



    cd = cal_cd(gt_xyz_norm, result_xyz_norm)
    hd = cal_hd(gt_xyz_norm, result_xyz_norm)
    emd = cal_emd(gt_xyz_norm, result_xyz_norm)
    jsd = cal_jsd(gt_xyz_norm, result_xyz_norm)

    # jsd = jsd_between_point_cloud_sets(gt_xyz_norm, result_xyz_norm)
    # p2f,_,_=pymesh.distance_to_mesh(pymesh.meshio.load_mesh(os.path.join('/data/IJS/IJS/dataset/pugeo_original_dataset/mesh','test_mesh',name[:-4]+'.off'),drop_zero_dim=False),result_xyz_norm,engine='auto')  # sketchfab
    if name == 'Mario.xyz':
        p2f,_,_=pymesh.distance_to_mesh(pymesh.meshio.load_mesh(os.path.join('/home/gpuadmin/IJS/Color_PUGeoNet/test_data/x4/gt_off',name[:-4]+'.off'),drop_zero_dim=False),result_xyz,engine='auto')
        p2f=np.sqrt(p2f).squeeze()
        p2f_list.append(p2f)


    cd_list.append(cd)
    hd_list.append(hd)
    emd_list.append(emd)
    jsd_list.append(jsd)

    

    # print(f"{name}'s mse : {mse}")

total_cd = sum(cd_list)/len(cd_list)
total_hd = sum(hd_list)/len(hd_list)
total_emd = sum(emd_list)/len(emd_list)
total_jsd = sum(jsd_list)/len(jsd_list)
# total_norm_emd = sum(normal_emd_list)/len(normal_emd_list)
total_p2f = np.concatenate(p2f_list, axis=0)

print(f"Total CD : {total_cd}")
print(f"Total HD : {total_hd}")
print(f"Total EMD : {total_emd}")
print(f"Total JSD : {total_jsd}")
# print(f"Total normal EMD : {total_norm_emd}")
print(f"Total Average P2F : {np.nanmean(total_p2f)}")
# print(f"Total STD P2F : {np.nanstd(total_p2f)}")




