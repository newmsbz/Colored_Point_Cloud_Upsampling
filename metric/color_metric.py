import open3d as o3d
import os
import numpy as np
import point_cloud_utils as pcu
from pc_util import normalize_point_cloud
from scipy.stats import wasserstein_distance
from scipy.spatial import distance
from skimage.metrics import peak_signal_noise_ratio


def cal_cd(gt, result):
    chamfer_dist = pcu.chamfer_distance(gt, result)
    return chamfer_dist


def cal_hd(gt, result):
    # Take a max of the one sided squared  distances to get the two sided Hausdorff distance
    hausdorff_dist = pcu.hausdorff_distance(gt, result)
    return hausdorff_dist

def cal_psnr(xyz_GT, xyz_Eval, clr_GT, clr_Eval):
    _, a_to_b, b_to_a = pcu.chamfer_distance(xyz_GT, xyz_Eval, return_index=True)
    # print(a_to_b)
    # print(clr_GT[a_to_b])

    # temp=GT[a_to_b]
    # temp2=Eval[a_to_b]
    # clr_GT_A = np.array(temp[:, 6:9]).astype(np.float32)
    # clr_Eval_A = np.array(temp2[:, 6:9]).astype(np.float32)

    clr_GT_A = clr_GT[a_to_b]
    clr_Eval_A = clr_Eval[a_to_b]

    PSNR_A=peak_signal_noise_ratio(clr_GT_A,clr_Eval_A)

    # temp=GT[b_to_a]
    # temp2=Eval[b_to_a]
    # clr_GT_B = np.array(temp[:, 6:9]).astype(np.float32)
    # clr_Eval_B = np.array(temp2[:, 6:9]).astype(np.float32)

    clr_GT_B = clr_GT[b_to_a]
    clr_Eval_B = clr_Eval[b_to_a]


    PSNR_B=peak_signal_noise_ratio(clr_GT_B,clr_Eval_B)

    return max(PSNR_A,PSNR_B)


gt_path = './test_data/x4/gt/'
# result_path = './test_data/x4/ours/'    # ours
# result_path = './test_data/ver25_600/'    # ours
# result_path = './test_data/x4/knn_based_result/k_3/'    # KNN=3
result_path = '/home/gpuadmin/IJS/Color_PUGeoNet/new_ver_save/ver27/x4/' # ver21


name_list = os.listdir(gt_path)
mse_list = []
mae_list = []
cd_list = []
hd_list = []
psnr_list = []

for name in name_list:
    print(name)
    gt_pcd_name = gt_path + name
    result_pcd_name = result_path + name
    
    gt_pcd = o3d.io.read_point_cloud(gt_pcd_name)
    result_pcd = o3d.io.read_point_cloud(result_pcd_name)

    gt_xyz = np.array(gt_pcd.points).astype(np.float32)
    gt_xyz_norm, _ , _ = normalize_point_cloud(gt_xyz)
    result_xyz = np.array(result_pcd.points).astype(np.float32)
    result_xyz_norm, _ , _ = normalize_point_cloud(result_xyz)


    gt_rgb = np.array(gt_pcd.colors).astype(np.float32)
    result_rgb = np.array(result_pcd.colors).astype(np.float32)

    mse = np.square(np.subtract(gt_rgb, result_rgb)).mean()
    mae = np.abs(gt_rgb-result_rgb).mean()
    cd = cal_cd(gt_xyz_norm, result_xyz_norm)
    hd = cal_hd(gt_xyz_norm, result_xyz_norm)
    psnr = cal_psnr(gt_xyz_norm, result_xyz_norm, gt_rgb, result_rgb)

    mse_list.append(mse)
    mae_list.append(mae)
    cd_list.append(cd)
    hd_list.append(hd)
    psnr_list.append(psnr)

    # print(f"{name}'s mse : {mse}")

total_mse = sum(mse_list)/len(mse_list)
total_mae = sum(mae_list)/len(mae_list)
total_cd = sum(cd_list)/len(cd_list)
total_hd = sum(hd_list)/len(hd_list)
total_psnr = sum(psnr_list)/len(psnr_list)

print(f"Total CD : {total_cd}")
print(f"Total HD : {total_hd}")
print(f"Total Color MSE : {total_mse}")
print(f"Total Color MaE : {total_mae}")
print(f"Total PSNR : {total_psnr}")

