import os
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    # print(x.shape)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature


def get_graph_color_feature(x, rgb, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = rgb.size()

    rgb = rgb.transpose(2,1).contiguous()
    feature = rgb.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    rgb = rgb.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - rgb, rgb), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature


def Nearest_Assignment(p_rgb, p_xyz, f_rgb, up_xyz, kn=1):

    batch_size = p_rgb.size(0)
    rgb_u = []
    feat_u = []

    for b in range(batch_size):
      
        dist = torch.cdist(up_xyz[b,:,:].T, p_xyz[b,:,:].T, p=2)
        distance1, knn1 = dist.topk(kn, largest=False)
        t_color = p_rgb[b,:,:].T
        f_color = f_rgb[b,:,:].T

        i_d = 1/ (distance1 + 1e-5).pow(2)
        sum = torch.sum(i_d, axis=1).reshape(-1,1)
        ft = i_d/sum
        n_color = torch.sum(torch.mul(t_color[knn1,:], ft.unsqueeze(-1).cuda()), dim=1)
        n_feat = torch.sum(torch.mul(f_color[knn1,:], ft.unsqueeze(-1).cuda()), dim=1)

        rgb_u.append(n_color)
        feat_u.append(n_feat)

    rgb_k = torch.stack(rgb_u).permute(0,2,1)
    feat_k = torch.stack(feat_u).permute(0,2,1)

    return rgb_k, feat_k


class pugeonet(nn.Module):
    def __init__(self, up_ratio, knn=15, fd=128, fD=1024):  # feature 개수가 2배로 늘었기 때문에 fd, fD값을 2배로 늘림
        super(pugeonet, self).__init__()
        self.knn = knn
        self.up_ratio = up_ratio
        self.feat_list = ["expand", "net_max_1", "net_mean_1",
                          "out3", "net_max_2", "net_mean_2",
                          "out5", "net_max_3", "net_mean_3",
                          "out7", "out8"]

        self.dgcnn_conv1 = nn.Sequential(nn.Conv2d(6, fd, kernel_size=1),
                                         nn.BatchNorm2d(fd),
                                         nn.LeakyReLU(negative_slope=0.2))
        self.dgcnn_conv2 = nn.Sequential(nn.Conv2d(fd, fd, kernel_size=1),
                                         nn.BatchNorm2d(fd),
                                         nn.LeakyReLU(negative_slope=0.2))
        self.dgcnn_conv3 = nn.Sequential(nn.Conv1d(fd + fd, fd, kernel_size=1),
                                         nn.BatchNorm1d(fd),
                                         nn.LeakyReLU(negative_slope=0.2))
        self.dgcnn_conv4 = nn.Sequential(nn.Conv2d(fd + fd, fd, kernel_size=1),
                                         nn.BatchNorm2d(fd),
                                         nn.LeakyReLU(negative_slope=0.2))
        self.dgcnn_conv5 = nn.Sequential(nn.Conv1d(fd + fd, fd, kernel_size=1),
                                         nn.BatchNorm1d(fd),
                                         nn.LeakyReLU(negative_slope=0.2))
        self.dgcnn_conv6 = nn.Sequential(nn.Conv2d(fd + fd, fd, kernel_size=1),
                                         nn.BatchNorm2d(fd),
                                         nn.LeakyReLU(negative_slope=0.2))
        self.dgcnn_conv7 = nn.Sequential(nn.Conv1d(fd + fd, fd, kernel_size=1),
                                         nn.BatchNorm1d(fd),
                                         nn.LeakyReLU(negative_slope=0.2))
        self.dgcnn_conv8 = nn.Sequential(nn.Conv1d(fd + fd + fd, fD, kernel_size=1),
                                         nn.BatchNorm1d(fD),
                                         nn.LeakyReLU(negative_slope=0.2))

        self.attention_conv1 = nn.Sequential(nn.Conv1d(fd * 9 + fD * 2, 256, kernel_size=1),
                                             nn.BatchNorm1d(256),
                                             nn.LeakyReLU(negative_slope=0.2))
        self.attention_conv2 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1),
                                             nn.BatchNorm1d(128),
                                             nn.LeakyReLU(negative_slope=0.2))
        self.attention_conv3 = nn.Sequential(nn.Conv1d(128, len(self.feat_list), kernel_size=1),
                                             nn.BatchNorm1d(len(self.feat_list)),
                                             nn.LeakyReLU(negative_slope=0.2))

        self.concat_conv = nn.Sequential(nn.Conv1d(fd * 9 + fD * 2, 256, kernel_size=1),    # original size : 256
                                         nn.BatchNorm1d(256),
                                         nn.LeakyReLU(negative_slope=0.2))
                                
        self.dg1 = nn.Sequential()

        self.uv_conv1 = nn.Sequential(nn.Conv1d(256, up_ratio * 2, kernel_size=1))

        self.patch_conv1 = nn.Sequential(nn.Conv1d(256, 9, kernel_size=1))

        self.normal_offset_conv1 = nn.Sequential(nn.Conv1d(256, up_ratio * 3, kernel_size=1))

        self.up_layer1 = nn.Sequential(nn.Conv2d(256 + 3, 128, kernel_size=1),
                                       nn.BatchNorm2d(128),
                                       nn.LeakyReLU(negative_slope=0.2))

        self.up_dg1 = nn.Sequential()
        self.up_layer2 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=1),
                                       nn.BatchNorm2d(128),
                                       nn.LeakyReLU(negative_slope=0.2))
        self.up_dg2 = nn.Sequential()

        self.fc_layer = nn.Conv2d(128, 1, kernel_size=1)

        self.clr_dgcnn_conv1 = nn.Sequential(nn.Conv2d(6, fd, kernel_size=1),
                                         nn.BatchNorm2d(fd),
                                         nn.LeakyReLU(negative_slope=0.2))

        self.clr_dgcnn_conv2 = nn.Sequential(nn.Conv2d(fd, fd, kernel_size=1),
                                         nn.BatchNorm2d(fd),
                                         nn.LeakyReLU(negative_slope=0.2))

        self.clr_dgcnn_conv3 = nn.Sequential(nn.Conv1d(fd + fd, fd, kernel_size=1),
                                         nn.BatchNorm1d(fd),
                                         nn.LeakyReLU(negative_slope=0.2))

        self.clr_dgcnn_conv4 = nn.Sequential(nn.Conv1d(fd, fD, kernel_size=1),
                                         nn.BatchNorm1d(fD),
                                         nn.LeakyReLU(negative_slope=0.2))

        self.clr_dgcnn_conv5 = nn.Sequential(nn.Conv1d(fD, 256, kernel_size=1),
                                         nn.BatchNorm1d(256),
                                         nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Sequential(nn.Conv1d(262, fD , kernel_size=1),
                                         nn.BatchNorm1d(fD),
                                         nn.LeakyReLU(negative_slope=0.2))
        self.linear2 = nn.Sequential(nn.Conv1d(fD , int(fD/2) , kernel_size=1),
                                         nn.BatchNorm1d(int(fD/2)),
                                         nn.LeakyReLU(negative_slope=0.2))
        self.linear3 = nn.Sequential(nn.Conv1d(int(fD/2),int(fD/4),kernel_size=1),
                                         nn.BatchNorm1d(int(fD/4)),
                                         nn.LeakyReLU(negative_slope=0.2))
        self.fc_clr_a =  nn.Conv1d(int(fD/4), 3, kernel_size=1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        batch_size = x.size(0)
        num_point = x.size(2)
        xyz, rgb = x[:,:3,:], x[:,3:,:]
        edge_feature = get_graph_feature(xyz, k=self.knn)

        out1 = self.dgcnn_conv1(edge_feature)
        out2 = self.dgcnn_conv2(out1)
        net_max_1 = out2.max(dim=-1, keepdim=False)[0]
        net_mean_1 = out2.mean(dim=-1, keepdim=False)

        out3 = self.dgcnn_conv3(torch.cat((net_max_1, net_mean_1), 1))

        edge_feature = get_graph_feature(out3, k=self.knn)
        out4 = self.dgcnn_conv4(edge_feature)

        net_max_2 = out4.max(dim=-1, keepdim=False)[0]
        net_mean_2 = out4.mean(dim=-1, keepdim=False)

        out5 = self.dgcnn_conv5(torch.cat((net_max_2, net_mean_2), 1))

        edge_feature = get_graph_feature(out5)
        out6 = self.dgcnn_conv6(edge_feature)

        net_max_3 = out6.max(dim=-1, keepdim=False)[0]
        net_mean_3 = out6.mean(dim=-1, keepdim=False)

        out7 = self.dgcnn_conv7(torch.cat((net_max_3, net_mean_3), dim=1))

        out8 = self.dgcnn_conv8(torch.cat((out3, out5, out7), 1))

        out_max = out8.max(dim=-1, keepdim=True)[0] 
        expand = out_max.repeat(1, 1, num_point)

        concat_unweight = torch.cat((expand,  
                                     net_max_1,  
                                     net_mean_1,
                                     out3,  
                                     net_max_2,
                                     net_mean_2,
                                     out5,  
                                     net_max_3,
                                     net_mean_3,
                                     out7,  
                                     out8  
                                     ), dim=1)  
        out_attention = self.attention_conv1(concat_unweight)
        out_attention = self.attention_conv2(out_attention)
        out_attention = self.attention_conv3(out_attention)  
        out_attention = out_attention.max(dim=-1, keepdim=False)[0]  
        out_attention = F.softmax(out_attention, dim=-1)  

        for i in range(len(self.feat_list)):
            tmp1 = out_attention[:, i]
            dim = eval('%s.size(1)' % self.feat_list[i])
            tmp2 = tmp1.unsqueeze(-1).repeat(1, dim)
            if i == 0:
                attention_weight = tmp2
            else:
                attention_weight = torch.cat((attention_weight, tmp2), axis=-1)
        attention_weight = attention_weight.unsqueeze(-1)
        concat = attention_weight * concat_unweight 
        concat = self.concat_conv(concat)

        concat = self.dg1(concat) 

        uv_2d = self.uv_conv1(concat)
        uv_2d = uv_2d.reshape(batch_size, self.up_ratio, 2, num_point)  
        uv_2d = torch.cat((uv_2d, torch.zeros((batch_size, self.up_ratio, 1, num_point)).to(x.device)),
                          dim=2)  

        affine_T = self.patch_conv1(concat)
        affine_T = affine_T.reshape(batch_size, 3, 3, num_point)  

        uv_3d = torch.matmul(uv_2d.permute(0, 3, 1, 2), affine_T.permute(0, 3, 1, 2))  
        uv_3d = uv_3d.permute(0, 2, 3, 1)  
        uv_3d = x[:,:3,:].unsqueeze(1).repeat(1, self.up_ratio, 1, 1) + uv_3d  

        uv_3d = uv_3d.transpose(1,2)   

        dense_normal_offset = self.normal_offset_conv1(concat)
        dense_normal_offset = dense_normal_offset.reshape(batch_size, self.up_ratio, 3, num_point)

        sparse_normal = torch.from_numpy(np.array([0, 0, 1]).astype(np.float32)).squeeze().reshape(1, 1, 3, 1).repeat(
            batch_size, 1, 1, num_point).to(x.device)

        sparse_normal = torch.matmul(sparse_normal.permute(0, 3, 1, 2), affine_T.permute(0, 3, 1, 2))
        sparse_normal = sparse_normal.permute(0, 2, 3, 1)
        sparse_normal = F.normalize(sparse_normal, dim=2) 

        dense_normal = sparse_normal.repeat(1, self.up_ratio, 1, 1, ) + dense_normal_offset
        dense_normal = F.normalize(dense_normal, dim=2)

        dense_normal=dense_normal.transpose(1,2).reshape(batch_size,3,-1)   

        grid = uv_3d

        concat_up=concat.unsqueeze(2).repeat(1,1,self.up_ratio,1) 
        coord_concat_up = torch.cat((concat_up, grid), axis=1) 
        coord_concat_up = self.up_layer1(coord_concat_up)
        coord_concat_up = self.up_dg1(coord_concat_up)
        coord_concat_up = self.up_layer2(coord_concat_up)
        coord_concat_up = self.up_dg2(coord_concat_up)

        coord_z = self.fc_layer(coord_concat_up)  #
        coord_z = torch.cat((torch.zeros_like(coord_z), torch.zeros_like(coord_z), coord_z), dim=1) 
        coord_z = torch.matmul(coord_z.permute(0, 3, 2, 1), affine_T.permute(0, 3, 1, 2))  
        coord = uv_3d + coord_z.permute(0, 3, 2, 1)     #(B,3,U,N)
        dense_coord=coord.reshape(batch_size,3,-1)

        color_feature = get_graph_color_feature(xyz, rgb, k=1)

        clr_out1 = self.clr_dgcnn_conv1(color_feature)
        clr_out2 = self.clr_dgcnn_conv2(clr_out1)
        clr_net_max_1 = clr_out2.max(dim=-1, keepdim=False)[0]
        clr_net_mean_1 = clr_out2.mean(dim=-1, keepdim=False)

        clr_out3 = self.clr_dgcnn_conv3(torch.cat((clr_net_max_1, clr_net_mean_1), 1))
        clr_out4 = self.clr_dgcnn_conv4(clr_out3)
        clr_out5 = self.clr_dgcnn_conv5(clr_out4)

        clr_kn_1, feat_kn_1 = Nearest_Assignment(rgb, xyz, clr_out5, dense_coord, kn=3)

        refined_clr_feat = torch.cat((clr_kn_1, feat_kn_1), dim=1)
        refined_clr_feat = torch.cat((refined_clr_feat, dense_normal), dim=1)

        refined_clr_feat = self.linear1(refined_clr_feat)
        refined_clr_feat = self.linear2(refined_clr_feat)
        refined_clr_feat = self.dropout(refined_clr_feat)
        refined_clr_feat = self.linear3(refined_clr_feat)
        color = self.fc_clr_a(refined_clr_feat)

        return dense_coord, dense_normal, sparse_normal.squeeze(1), color

