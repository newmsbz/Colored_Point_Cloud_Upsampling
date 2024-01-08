import os
import torch
from dataloader_pytorch import dataset_patch
import numpy as np
import argparse
import model
import logging
from glob import glob
from pc_util import normalize_point_cloud, farthest_point_sample, group_points
from datetime import datetime
from tqdm import tqdm, trange
from torch.optim.lr_scheduler import CosineAnnealingLR
from skimage.color import rgb2hsv
import math
import torch.nn.functional as f

def log_string(out_str):
    global LOG_FOUT
    LOG_FOUT.write(out_str)
    LOG_FOUT.flush()


def eval_patches(xyz, arg, model):
    centroids = farthest_point_sample(xyz, arg.num_patch)

    patches = group_points(xyz, centroids, arg.num_point)

    normalized_patches, patch_centroid, furthest_distance = normalize_point_cloud(patches)

    dense_patches_list = []
    dense_normal_list = []
    sparse_normal_list = []

    for i in range(normalized_patches.shape[0]):
        dense_patches, dense_normal, sparse_normal = model(
            torch.from_numpy(normalized_patches[i]).unsqueeze(0).transpose(1, 2), 1)
        dense_patches_list.append(dense_patches.transpose(1, 2).cpu().numpy())
        dense_normal_list.append(dense_normal.transpose(1, 2).cpu().numpy())
        sparse_normal_list.append(sparse_normal.transpose(1, 2).cpu().numpy())

    gen_ddense_xyz = np.concatenate(dense_patches_list, axis=0)
    gen_ddense_xyz = (gen_ddense_xyz * furthest_distance) + patch_centroid
    gen_ddense_normal = np.concatenate(dense_normal_list, axis=0)

    return np.reshape(gen_ddense_xyz, (-1, 3)), np.reshape(gen_ddense_normal, (-1, 3))


if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', default='/home/gpuadmin/IJS/Color_PUGeoNet/data/patches', help='train or test')  # Train Patch's path
    parser.add_argument('--phase', default='train', help='train or test')
    parser.add_argument('--gpu', default='0', help='which gpu to use')
    parser.add_argument('--up_ratio', type=int, default=4, help='Upsampling Ratio')  #
    parser.add_argument('--model', default='model_pugeo', help='Model for upsampling')
    parser.add_argument('--num_point', type=int, default=64, help='Point Number')

    parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training')
    parser.add_argument('--max_epoch', type=int, default=300, help='Epoch to run')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--min_lr', type=float, default=0.00001)
    parser.add_argument('--reg_normal1', type=float, default=0.1)
    parser.add_argument('--reg_normal2', type=float, default=0.1)
    parser.add_argument('--jitter_sigma', type=float, default=0.01)
    parser.add_argument('--jitter_max', type=float, default=0.03)

    # for phase test
    parser.add_argument('--pretrained', default='', help='Model stored')
    parser.add_argument('--eval_xyz', default='D:\\PUGEO\\test_5000', help='Folder to evaluate')    # test point cloud(.xyz)'s path
    parser.add_argument('--num_shape_point', type=int, default=1000, help='Point Number per shape')
    parser.add_argument('--patch_num_ratio', type=int, default=3, help='Number of points covered by patch')
    arg = parser.parse_args()
    arg.num_patch = int(arg.num_shape_point / arg.num_point * arg.patch_num_ratio)
    arg.log_dir='.log/_x%d'%arg.up_ratio
    try:
        os.mkdir(arg.log_dir)
    except:
        pass
    global LOG_FOUT
    LOG_FOUT = open(os.path.join(arg.log_dir, 'log.txt'), 'w')
    LOG_FOUT.write(str(datetime.now()) + '\n')
    LOG_FOUT.write(os.path.abspath(__file__) + '\n')
    LOG_FOUT.write(str(arg) + '\n')

    dataset = dataset_patch(arg.up_ratio,arg.data_path,jitter=False,jitter_max=arg.jitter_max, jitter_sigma=arg.jitter_sigma,mode='train')
    dataloader=torch.utils.data.DataLoader(dataset,batch_size=arg.batch_size,shuffle=True,drop_last=True,num_workers=8)


    model=model.pugeonet(up_ratio=arg.up_ratio,knn=15)  # default knn : 30
    model=model.cuda()

    learnable_params = filter(lambda p: p.requires_grad, model.parameters())

    current_lr=arg.learning_rate

    optimizer=torch.optim.Adam(learnable_params,lr=current_lr)
    scheduler = CosineAnnealingLR(optimizer, arg.max_epoch, eta_min=current_lr)

    for epoch in range(arg.max_epoch):
        scheduler.step()
        loss_sum_all=[]
        loss_sum_dense_cd = []
        loss_sum_dense_normal = []
        loss_sum_sparse_normal = []
        loss_sum_color = []


        for data in tqdm(dataloader,desc='epoch %d'%epoch):
            input_sparse_xyz,input_sparse_normal,input_rgb,gt_dense_xyz,gt_dense_normal,gt_rgb,input_r=data
            input_sparse_xyz = input_sparse_xyz.cuda()
            input_sparse_normal = input_sparse_normal.cuda()
            input_rgb = input_rgb.cuda()
            gt_dense_xyz = gt_dense_xyz.cuda()
            gt_dense_normal = gt_dense_normal.cuda()
            gt_rgb = gt_rgb.cuda()
            input_r = input_r.cuda()
            input_xyz_rgb = torch.cat((input_sparse_xyz, input_rgb), dim=1)
            
            optimizer.zero_grad()

            model.train()

            gen_dense_xyz,gen_dense_normal,gen_sparse_normal,gen_rgb=model(input_xyz_rgb)

            loss_dense_cd,cd_idx1,cd_idx2=loss.cd_loss(gen_dense_xyz,gt_dense_xyz,input_r)
            loss_dense_normal=loss.abs_dense_normal_loss(gen_dense_normal,gt_dense_normal,cd_idx1,cd_idx2,input_r)
            loss_sparse_normal=loss.abs_sparse_normal_loss(gen_sparse_normal,input_sparse_normal,input_r)

            loss_color_close = 0
            for b in range(16):
                dist = torch.cdist(gen_dense_xyz[b,:,:].T, input_sparse_xyz[b,:,:].T, p=2)
                distance1, knn1 = dist.topk(1, largest=False)
                norm_distance1 = f.normalize(distance1, dim=0)
                sparse_color = input_rgb[b,:,:].T
                dense_color = gen_rgb[b,:,:].T

                closest_color = sparse_color[knn1, :]
                dist_weight = torch.ones_like(norm_distance1) + norm_distance1

                refined_closest_color_loss = loss.weighted_color_mse_loss(dense_color, closest_color.squeeze(), dist_weight)

                loss_color_close += refined_closest_color_loss

            loss_color = loss_color_close + loss.color_mae_loss(gen_rgb, gt_rgb)   
            loss_all = (100 * loss_dense_cd + arg.reg_normal1 * loss_dense_normal + arg.reg_normal2 * loss_sparse_normal) + loss_color

            loss_all.backward()
            optimizer.step()

            loss_sum_all.append(loss_all.detach().cpu().numpy())
            loss_sum_dense_cd.append(loss_dense_cd.detach().cpu().numpy())
            loss_sum_dense_normal.append(loss_dense_normal.detach().cpu().numpy())
            loss_sum_sparse_normal.append(loss_sparse_normal.detach().cpu().numpy())
            loss_sum_color.append(loss_color.detach().cpu().numpy())

        loss_sum_all = np.asarray(loss_sum_all)
        loss_sum_dense_cd = np.asarray(loss_sum_dense_cd)
        loss_sum_dense_normal = np.asarray(loss_sum_dense_normal)
        loss_sum_sparse_normal = np.asarray(loss_sum_sparse_normal)
        loss_sum_color = np.asarray(loss_sum_color)
        log_string('epoch: %d total loss: %f, cd: %f, dense normal: %f, sparse normal: %f, color: %f\n' % (
                    epoch, round(loss_sum_all.mean(), 7), round(loss_sum_dense_cd.mean(), 7), round(loss_sum_dense_normal.mean(), 7),
                    round(loss_sum_sparse_normal.mean(), 7), round(loss_sum_color.mean(), 7)))
        
    torch.save(model.state_dict(), os.path.join(arg.log_dir, 'model.t7'))
