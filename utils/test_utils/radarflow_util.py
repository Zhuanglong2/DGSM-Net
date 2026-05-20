import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np
# from lib import pointnet2_utils as pointutils
from pointnet2_ops import pointnet2_utils as pointutils
from .pt_mamba import MixerModel, Block_mamba

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    dist = torch.maximum(dist,torch.zeros(dist.size()).cuda())
    return dist


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def index_points_group(points, knn_idx):
    """
    Input:
        points: input points data, [B, N, C]
        knn_idx: sample index data, [B, N, K]
    Return:
        new_points:, indexed points data, [B, N, K, C]
    """
    points_flipped = points.permute(0, 2, 1).contiguous()
    new_points = pointutils.grouping_operation(points_flipped, knn_idx.int()).permute(0, 2, 3, 1)

    return new_points

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)
    return group_idx

class MultiScaleEncoder(nn.Module):
    def __init__(self, radius, nsample, in_channel, mlp, mlp2):
        # ... (初始化代码与之前保持一致) ...
        super(MultiScaleEncoder, self).__init__()

        self.ms_ls = nn.ModuleList()
        num_sas = len(radius)
        for l in range(num_sas):
            # PointLocalFeature 的初始化保持不变
            self.ms_ls.append(PointLocalFeature(radius[l], \
                                                nsample[l], in_channel=in_channel, mlp=mlp, mlp2=mlp2))

    def forward(self, xyz, features):

        # 初始化两个零张量用于收集特征
        new_features = torch.zeros(0).cuda()  # 收集最终融合特征
        vr_features_ms = torch.zeros(0).cuda()  # 收集独立的 vr 编码特征

        for i, sa in enumerate(self.ms_ls):
            # sa() 现在返回两个张量
            final_fused_f, vr_f_scale = sa(xyz, features)

            # 拼接最终融合特征
            new_features = torch.cat((new_features, final_fused_f), dim=1)

            # 拼接独立的 vr 编码特征 (所有尺度)
            vr_features_ms = torch.cat((vr_features_ms, vr_f_scale), dim=1)

        # 返回最终的多尺度融合特征 和 多尺度的 vr 特征
        return new_features, vr_features_ms  # <-- 返回两个张量


class PointLocalFeature(nn.Module):
    def __init__(self, radius, nsample, in_channel, mlp, mlp2):
        super(PointLocalFeature, self).__init__()
        self.radius = radius
        self.nsample = nsample

        # --- [1] 原有 MLP 初始化 ---
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()

        last_channel = in_channel + 3
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1, bias=False))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        # --- [2] V_r 特征子分支 MLP 初始化 ---
        self.vr_output_dim = 16
        self.vr_sub_mlp = nn.Sequential(
            nn.Conv2d(1, self.vr_output_dim, 1, bias=False),  # 仅一层 Conv2D 投影
            nn.BatchNorm2d(self.vr_output_dim),
            nn.ReLU()
        )

        # --- [3] 最终聚合 MLP 初始化 (新的 last_channel 已包含 vr_output_dim) ---
        last_channel = mlp[-1] + self.vr_output_dim  # 融合后的维度
        self.mlp2_convs = nn.ModuleList()
        self.mlp2_bns = nn.ModuleList()
        for out_channel in mlp2:
            self.mlp2_convs.append(nn.Conv2d(last_channel, out_channel, 1, bias=False))
            self.mlp2_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        self.queryandgroup = pointutils.QueryAndGroup(radius, nsample)

    def forward(self, xyz, points):
        device = xyz.device
        B, C, N = xyz.shape
        xyz_t = xyz.permute(0, 2, 1).contiguous()

        grouped_points = self.queryandgroup(xyz_t, xyz_t, points)  # [B, C+3, N, K]

        # 1. V_r 分支提取 (假设 vr 在特征通道的索引 0，即 grouped_points 的索引 3)
        vr_raw_grouped = grouped_points[:, 3:4, :, :]
        vr_sub_features = self.vr_sub_mlp(vr_raw_grouped)  # [B, vr_output_dim, N, K]

        # 2. 主分支提取
        new_points = grouped_points
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))  # [B, mlp[-1], N, K]

        # 3. 提取独立的 V_r 特征 (Mean-Pooling 聚合)
        # 这是该尺度下纯粹的 vr 编码
        vr_feature_scale = torch.mean(vr_sub_features, -1)[0]  # [B, vr_output_dim, N]

        # 4. 融合 (用于主输出)
        fused_points = torch.cat((new_points, vr_sub_features), dim=1)  # [B, C_fused, N, K]

        # 5. Max-Pooling 聚合（针对融合后的特征进行）
        new_points = torch.max(fused_points, -1)[0].unsqueeze(2)  # [B, C_fused, N, 1]

        # 6. 最终 MLP (mlp2)
        for i, conv in enumerate(self.mlp2_convs):
            bn = self.mlp2_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        final_fused_feature = new_points.squeeze(2)  # [B, mlp2[-1], N]

        # 返回最终融合特征 和 独立的 vr 特征
        return final_fused_feature, vr_feature_scale  # <-- 返回两个张量

class FlowHead(nn.Module):
    def __init__(self, in_channel, mlp):
        super(FlowHead, self).__init__()
        self.sf_mlp = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.sf_mlp.append(nn.Sequential(nn.Conv2d(last_channel, out_channel, 1, bias=False),
                                                 nn.BatchNorm2d(out_channel),
                                                 nn.ReLU(inplace=False)))
            last_channel = out_channel
            
        self.conv2 = nn.Conv2d(mlp[-1], 3, 1, bias=False)
        
    def forward(self, feat):
 
        feat = feat.unsqueeze(3)
        for conv in self.sf_mlp:
            feat = conv(feat)
        
        output = self.conv2(feat)
        
        return output.squeeze(3)

class MotionHead(nn.Module):
    def __init__(self, in_channel, mlp):
        super(MotionHead, self).__init__()
        self.sf_mlp = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.sf_mlp.append(nn.Sequential(nn.Conv2d(last_channel, out_channel, 1, bias=False),
                                                 nn.BatchNorm2d(out_channel),
                                                 nn.ReLU(inplace=False)))
            last_channel = out_channel
            
        self.conv2 = nn.Conv2d(mlp[-1], 1, 1, bias=False)
        self.m = nn.Sigmoid()
        
    def forward(self, feat):
 
        feat = feat.unsqueeze(3)
        for conv in self.sf_mlp:
            feat = conv(feat)
        
        output = self.m(self.conv2(feat))
        
        return output.squeeze(3)

class OdmHead(nn.Module):
    def __init__(self, in_channel, out_channel_final, mlp):
        super(OdmHead, self).__init__()
        self.sf_mlp = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.sf_mlp.append(nn.Sequential(nn.Conv2d(last_channel, out_channel, 1, bias=False),
                                             nn.BatchNorm2d(out_channel),
                                             nn.ReLU(inplace=False)))
            last_channel = out_channel

        self.conv2 = nn.Conv2d(mlp[-1], out_channel_final, 1, bias=False)
        self.m = nn.Sigmoid()

    def forward(self, feat):

        feat = feat.unsqueeze(3)
        for conv in self.sf_mlp:
            feat = conv(feat)

        output = self.m(self.conv2(feat))

        return output.squeeze(3)
    
class WeightNet(nn.Module):

    def __init__(self, in_channel, out_channel, hidden_unit = [8, 8], bn = False):
        super(WeightNet, self).__init__()

        self.bn = bn
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        if hidden_unit is None or len(hidden_unit) == 0:
            self.mlp_convs.append(nn.Conv2d(in_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
        else:
            self.mlp_convs.append(nn.Conv2d(in_channel, hidden_unit[0], 1))
            self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[0]))
            for i in range(1, len(hidden_unit)):
                self.mlp_convs.append(nn.Conv2d(hidden_unit[i - 1], hidden_unit[i], 1))
                self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[i]))
            self.mlp_convs.append(nn.Conv2d(hidden_unit[-1], out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
        
    def forward(self, localized_xyz):
        #xyz : BxCxKxN

        weights = localized_xyz
        for i, conv in enumerate(self.mlp_convs):
            if self.bn:
                bn = self.mlp_bns[i]
                weights =  F.relu(bn(conv(weights)))
            else:
                weights = F.relu(conv(weights))

        return weights


class FlowDecoder(nn.Module):
    def __init__(self, fc_inch):
        super(FlowDecoder, self).__init__()
        ## multi-scale flow embeddings propogation
        # different scale share the same mlps hyper-parameters
        ep_radius = [2.0, 4.0, 8.0, 16.0]
        ep_nsamples = [4, 8, 16, 32]
        ep_inch = fc_inch * 2 + 3
        ep_mlps = [fc_inch, int(fc_inch/2), int(fc_inch/8)]
        ep_mlp2s = [int(fc_inch/8), int(fc_inch/8), int(fc_inch/8)]
        num_eps = len(ep_radius)
        self.mse = MultiScaleEncoder(ep_radius, ep_nsamples, in_channel=ep_inch, \
                                         mlp = ep_mlps, mlp2 = ep_mlp2s)
        ## scene flow predictor
        sf_inch = num_eps * ep_mlp2s[-1]*2
        sf_mlps = [int(sf_inch/2), int(sf_inch/4), int(sf_inch/8)]
        self.fp = FlowPredictor(in_channel=sf_inch, mlp=sf_mlps)
        
    def forward(self, pc1, feature1, pc1_features, cor_features):
        
        embeddings = torch.cat((feature1, pc1_features, cor_features),dim=1)
        ## multi-scale flow embeddings propogation
        prop_features = self.mse(pc1,embeddings)
        gfeat = torch.max(prop_features,-1)[0].unsqueeze(2).expand(prop_features.size()[0],prop_features.size()[1],pc1.size()[2])
        final_features = torch.cat((prop_features, gfeat),dim=1)
        
        ## initial scene flow prediction
        output = self.fp(final_features)
        
        return output
    

class Decoder(nn.Module):
    def __init__(self, fc_inch):
        super(Decoder, self).__init__()
        ## multi-scale flow embeddings propogation
        # different scale share the same mlps hyper-parameters
        ep_radius = [2.0, 4.0, 8.0, 16.0]
        ep_nsamples = [4, 8, 16, 32]
        ep_inch = fc_inch * 2 + 3
        ep_mlps = [fc_inch, int(fc_inch/2), int(fc_inch/8)]
        ep_mlp2s = [int(fc_inch/8), int(fc_inch/8), int(fc_inch/8)]
        num_eps = len(ep_radius)
        self.mse = MultiScaleEncoder(ep_radius, ep_nsamples, in_channel=ep_inch, \
                                         mlp = ep_mlps, mlp2 = ep_mlp2s)
        ## scene flow predictor
        sf_inch = num_eps * ep_mlp2s[-1]*2
        sf_mlps = [int(sf_inch/2), int(sf_inch/4), int(sf_inch/8)]
        self.fp = FlowPredictor(in_channel=sf_inch, mlp=sf_mlps)
        self.mp = MotionPredictor(in_channel=sf_inch, mlp=sf_mlps)
        
        
    def forward(self, pc1, feature1, pc1_features, cor_features):
        
        embeddings = torch.cat((feature1, pc1_features, cor_features),dim=1)
        ## multi-scale flow embeddings propogation
        prop_features = self.mse(pc1,embeddings)
        gfeat = torch.max(prop_features,-1)[0].unsqueeze(2).expand(prop_features.size()[0],prop_features.size()[1],pc1.size()[2])
        final_features = torch.cat((prop_features, gfeat),dim=1)
        
        ## initial scene flow prediction
        output = self.fp(final_features)
        static_cls = self.mp(final_features)
        
        return output, static_cls
    
        
class FlowPredictor(nn.Module):
    def __init__(self, in_channel, mlp):
        super(FlowPredictor, self).__init__()
        self.sf_mlp = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.sf_mlp.append(nn.Sequential(nn.Conv2d(last_channel, out_channel, 1, bias=False),
                                                 nn.BatchNorm2d(out_channel),
                                                 nn.ReLU(inplace=False)))
            last_channel = out_channel
            
        self.conv2 = nn.Conv2d(mlp[-1], 3, 1, bias=False)
        
    def forward(self, feat):
 
        feat = feat.unsqueeze(3)
        for conv in self.sf_mlp:
            feat = conv(feat)
        
        output = self.conv2(feat)
        
        return output.squeeze(3)

class MotionPredictor(nn.Module):
    def __init__(self, in_channel, mlp):
        super(MotionPredictor, self).__init__()
        self.sf_mlp = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.sf_mlp.append(nn.Sequential(nn.Conv2d(last_channel, out_channel, 1, bias=False),
                                                 nn.BatchNorm2d(out_channel),
                                                 nn.ReLU(inplace=False)))
            last_channel = out_channel
            
        self.conv2 = nn.Conv2d(mlp[-1], 1, 1, bias=False)
        self.m = nn.Sigmoid()
        
    def forward(self, feat):
 
        feat = feat.unsqueeze(3)
        for conv in self.sf_mlp:
            feat = conv(feat)
        
        output = self.m(self.conv2(feat))
        
        return output.squeeze(3)

class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel
        self.weightnet = WeightNet(1, last_channel)

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(points2 * weight, dim=2)

        if points1 is not None:
            new_points = torch.cat([points1, interpolated_points.unsqueeze(1)], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points.permute(0, 2, 1)

#mamba
class NDTFeatureCorrelator(nn.Module):
    def __init__(self, nsample=16, out_channel=512, upsample_point=384, use_leaky=True):
        """
        nsample: 每个点邻域采样数
        out_channel: 输出特征维度
        upsample_point: 上采样输出点数
        """
        super(NDTFeatureCorrelator, self).__init__()
        self.nsample = nsample
        self.upsample_point = upsample_point
        self.out_channel = out_channel
        self.relu = nn.LeakyReLU(0.1, inplace=True) if use_leaky else nn.ReLU(inplace=True)

        # 固定结构，不再动态初始化
        self.mlp = nn.Sequential(
            nn.Conv1d(512 + 1, 256, kernel_size=1),
            nn.BatchNorm1d(256),
            self.relu,
            nn.Conv1d(256, out_channel, kernel_size=1),
            nn.BatchNorm1d(out_channel),
            self.relu
        )

    def forward(self, xyz1, xyz2, points1, points2):
        """
        xyz1, xyz2: Bx3xN
        points1, points2: BxDxN (雷达融合后的物理特征)
        """
        B, _, N1 = xyz1.shape
        _, D, _ = points1.shape

        # 转为 BxNxC/D
        xyz1_t = xyz1.permute(0, 2, 1)
        xyz2_t = xyz2.permute(0, 2, 1)
        points1_t = points1.permute(0, 2, 1)
        points2_t = points2.permute(0, 2, 1)

        # --- Step 1. 局部邻域 ---
        knn_idx = knn_point(self.nsample, xyz2_t, xyz1_t)
        neighbor_xyz = index_points_group(xyz2_t, knn_idx)       # B x N1 x nsample x 3
        neighbor_feat = index_points_group(points2_t, knn_idx)   # B x N1 x nsample x D

        # --- Step 2. NDT统计特征 ---
        mu = neighbor_xyz.mean(dim=2, keepdim=True)              # B x N1 x 1 x 3
        diff = neighbor_xyz - mu
        cov = torch.matmul(diff.transpose(-1, -2), diff) / (self.nsample + 1e-6)  # B x N1 x 3 x 3
        cov += 1e-5 * torch.eye(3, device=cov.device).unsqueeze(0).unsqueeze(0)

        # --- Step 3. Mahalanobis距离 ---
        diff_xyz = xyz1_t.unsqueeze(2) - mu                      # B x N1 x 1 x 3
        cov_inv = torch.linalg.pinv(cov)                         # B x N1 x 3 x 3
        mahalanobis = torch.matmul(torch.matmul(diff_xyz, cov_inv), diff_xyz.transpose(-1, -2))
        # spatial_likelihood = torch.exp(-mahalanobis.squeeze(-1).squeeze(-1))  # B x N1
        spatial_likelihood = F.softmax(-mahalanobis.squeeze(-1).squeeze(-1), dim=-1)

        # --- Step 4. 特征相似度 ---
        feat_diff = neighbor_feat - points1_t.unsqueeze(2)
        feature_likelihood = torch.exp(-torch.mean(feat_diff ** 2, dim=-1)).mean(dim=-1)  # B x N1
        # feature_likelihood = torch.sigmoid(-torch.sum(feat_diff ** 2, dim=-1) / D)

        # --- Step 5. 融合likelihood ---
        likelihood = (spatial_likelihood + feature_likelihood) / 2.0
        likelihood = torch.clamp(likelihood, min=1e-6, max=1.0)

        # --- Step 6. 融合到特征通道 ---
        # 原始4D雷达特征中包含 range, doppler, azimuth, elevation 四个维度
        # likelihood 反映“几何可信度”，融合后让特征包含物理与空间统计信息
        new_points = torch.cat([points1, likelihood.unsqueeze(1)], dim=1)  # B x (D+1) x N1

        # --- Step 7. 对齐维度并激活 ---
        new_points = self.mlp(new_points)

        return new_points


class FeatureCorrelatorNDT(nn.Module):
    def __init__(self, nsample, in_channel, mlp, bn=False, use_leaky=True, use_ndt=True):
        super(FeatureCorrelatorNDT, self).__init__()
        self.nsample = nsample
        self.bn = bn
        self.use_ndt = use_ndt
        self.in_channel = in_channel

        # ---------------- MLP ----------------
        # 输入 concat 维度: g1, g2, diff, direction -> 3*in_channel + 3
        # mlp_in = in_channel * 3 + 3
        mlp_in = 1539
        layers = []
        last = mlp_in
        for outc in mlp:
            layers.append(nn.Conv2d(last, outc, 1, bias=False))
            if bn:
                layers.append(nn.BatchNorm2d(outc))
            layers.append(nn.LeakyReLU(0.1, inplace=True) if use_leaky else nn.ReLU(inplace=True))
            last = outc
        self.mlp = nn.Sequential(*layers)
        self.mlp_out_channels = last

        # ---------------- 投影层 ----------------
        # 输入特征维度不匹配时的投影
        self.proj1 = nn.Conv1d(in_channel, self.mlp_out_channels, 1, bias=False)
        self.proj2 = nn.Conv1d(in_channel, self.mlp_out_channels, 1, bias=False)

        # ---------------- NDT 模块 ----------------
        if use_ndt:
            self.ndt_module = NDTFeatureCorrelator(
                nsample=self.nsample, out_channel=512, upsample_point=320
            )
            self.ndt_gate = nn.Sequential(
                nn.Conv1d(self.mlp_out_channels, self.mlp_out_channels, 1),
                nn.BatchNorm1d(self.mlp_out_channels) if bn else nn.Identity(),
                nn.LeakyReLU(0.1, inplace=True) if use_leaky else nn.ReLU(inplace=True),
                nn.Conv1d(self.mlp_out_channels, self.mlp_out_channels, 1),
                nn.Sigmoid()
            )
            # NDT 投影层（避免 forward 动态创建）
            self.proj_ndt = nn.Conv1d(512, self.mlp_out_channels, 1, bias=False)

        # ---------------- 几何权重 ----------------
        self.weightnet_dir = WeightNet(3, self.mlp_out_channels)
        self.weightnet_feat = WeightNet(1, self.mlp_out_channels)

        # ---------------- Patch-to-Patch 模块 ----------------
        self.blocks = MixerModel(d_model=512, n_layer=1, rms_norm=False)
        self.propagation = PointNetFeaturePropagation(in_channel=1024, mlp=[512])

        self.act = nn.LeakyReLU(0.1, inplace=True) if use_leaky else nn.ReLU(inplace=True)

    # -------- Mamba 模块 --------
    def mamba(self, xyz, feat, blocks):
        B, M, _ = xyz.shape
        _, _, Cfeat = feat.shape

        idx_x = xyz[:, :, 0].argsort(dim=-1)
        idx_y = xyz[:, :, 1].argsort(dim=-1)
        idx_z = xyz[:, :, 2].argsort(dim=-1)

        def gather_feat(idx):
            idx_exp = idx.unsqueeze(-1).expand(-1, -1, Cfeat)
            return feat.gather(1, idx_exp), xyz.gather(1, idx.unsqueeze(-1).expand(-1, -1, 3))

        tok_x, cx = gather_feat(idx_x)
        tok_y, cy = gather_feat(idx_y)
        tok_z, cz = gather_feat(idx_z)

        tokens = torch.cat([tok_x, tok_y, tok_z], dim=1)
        centers = torch.cat([cx, cy, cz], dim=1)

        tokens_out = blocks(tokens)
        return tokens_out, centers

    # -------- 前向传播 --------
    def forward(self, xyz1, xyz2, points1, points2, *args, **kwargs):

        # ----- NDT 提取 -----
        if self.use_ndt:
            ndt_feat = self.ndt_module(xyz1, xyz2, points1, points2)

        def _ensure_xyz(x):
            if x.dim() == 3 and x.shape[1] == 3:
                return x.permute(0, 2, 1)
            return x

        def _ensure_points(p):
            if p.dim() == 3 and p.shape[1] != 3:
                return p.permute(0, 2, 1)
            return p

        xyz1, xyz2 = _ensure_xyz(xyz1), _ensure_xyz(xyz2)
        points1, points2 = _ensure_points(points1), _ensure_points(points2)

        B, N1, _ = xyz1.shape

        # ----- Point-to-Patch -----
        knn_idx = knn_point(self.nsample, xyz2, xyz1)
        neighbor_xyz = index_points_group(xyz2, knn_idx)
        neighbor_feat = index_points_group(points2, knn_idx)

        group1 = points1.unsqueeze(2).expand(-1, -1, self.nsample, -1)
        group2 = neighbor_feat
        direction = neighbor_xyz - xyz1.unsqueeze(2)

        D1, D2 = group1.shape[-1], group2.shape[-1]
        if D1 != D2:
            g1 = self.proj1(group1.reshape(-1, D1).unsqueeze(-1)).squeeze(-1).view(B, N1, self.nsample, -1)
            g2 = self.proj2(group2.reshape(-1, D2).unsqueeze(-1)).squeeze(-1).view(B, N1, self.nsample, -1)
        else:
            g1, g2 = group1, group2

        diff = g2 - g1
        concat = torch.cat([g1, g2, diff, direction], dim=-1).permute(0, 3, 2, 1)
        new_points = self.mlp(concat)

        weights_geo = self.weightnet_dir(direction.permute(0, 3, 2, 1))
        sim = torch.mean(F.normalize(g1, dim=-1) * F.normalize(g2, dim=-1), dim=-1, keepdim=True)
        weights_feat = self.weightnet_feat(sim.permute(0, 3, 2, 1))

        weighted_geo = (weights_geo * new_points).sum(dim=2)
        weighted_feat = (weights_feat * new_points).sum(dim=2)
        point_to_patch_cost = weighted_geo + weighted_feat

        # ----- NDT 融合 -----
        if self.use_ndt:
            ndt_proj = self.proj_ndt(ndt_feat)
            gate = self.ndt_gate(point_to_patch_cost)
            point_to_patch_cost = point_to_patch_cost + gate * ndt_proj[:, :, :point_to_patch_cost.shape[2]]

        # ----- Patch-to-Patch -----
        knn_idx2 = knn_point(self.nsample, xyz1, xyz1)
        neighbor_xyz2 = index_points_group(xyz1, knn_idx2)
        ptp_feat = point_to_patch_cost.permute(0, 2, 1)
        neighbor_feat2 = index_points_group(ptp_feat, knn_idx2)

        B, N1, ns, Cpt = neighbor_feat2.shape
        new_patch_feat, new_xyz1 = self.mamba(
            neighbor_xyz2.reshape(B * N1, ns, 3),
            neighbor_feat2.reshape(B * N1, ns, Cpt),
            self.blocks
        )

        src_xyz = xyz1.reshape(B * N1, 1, 3)
        src_feat = point_to_patch_cost.permute(0, 2, 1).reshape(B * N1, 1, -1)

        patch_to_patch_cost = self.propagation(src_xyz, new_xyz1, src_feat, new_patch_feat)
        patch_to_patch_cost = patch_to_patch_cost.view(B, N1, -1).permute(0, 2, 1)

        return patch_to_patch_cost
####################################################################


class FeatureCorrelator(nn.Module):
    def __init__(self, nsample, in_channel, mlp, bn = False, use_leaky = True):
        super(FeatureCorrelator, self).__init__()
        self.nsample = nsample
        self.bn = bn
        self.mlp_convs = nn.ModuleList()
        self.mlp_convs0 = nn.ModuleList()
        if bn:
            self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            if bn:
                self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        if bn:
            self.mlp_bns0 = nn.ModuleList()
        last_channel = 515 * 3
        for out_channel in mlp:
            self.mlp_convs0.append(nn.Conv2d(last_channel, out_channel, 1))
            if bn:
                self.mlp_bns0.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        self.weightnet0 = WeightNet(3, last_channel)
        self.weightnet0_feat = WeightNet(1, last_channel)
        self.weightnet1 = WeightNet(3, last_channel)
        self.weightnet1_feat = WeightNet(1, last_channel)
        self.weightnet2 = WeightNet(3, last_channel)
        self.weightnet3 = WeightNet(3, last_channel)

        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(0.1, inplace=True)
        self.affine_alpha = nn.Parameter(torch.ones([1, 1, 1, last_channel+3]))
        self.affine_beta = nn.Parameter(torch.zeros([1, 1, 1, last_channel+3]))

        self.affine_alpha_ = nn.Parameter(torch.ones([1, 1, 1, last_channel+3]))
        self.affine_beta_ = nn.Parameter(torch.zeros([1, 1, 1, last_channel+3]))
        self.alph = nn.Parameter(torch.tensor([0.5]), requires_grad=True)

        self.blocks = MixerModel(d_model=512,
                                 n_layer=1,
                                 rms_norm=False)
        self.blocks_ = MixerModel(d_model=512,
                                  n_layer=1,
                                  rms_norm=False)
        self.propagation = PointNetFeaturePropagation(in_channel=1024, mlp=[512])
        self.propagation_ = PointNetFeaturePropagation(in_channel=1024, mlp=[512])

    def mamba(self, xyz, feature, blocks):
        # reordering strategy
        center_x = xyz[:, :, 0].argsort(dim=-1)[:, :, None]
        center_y = xyz[:, :, 1].argsort(dim=-1)[:, :, None]
        center_z = xyz[:, :, 2].argsort(dim=-1)[:, :, None]
        group_input_tokens_x = feature.gather(dim=1, index=torch.tile(center_x, (
            1, 1, feature.shape[-1])))
        group_input_tokens_y = feature.gather(dim=1, index=torch.tile(center_y, (
            1, 1, feature.shape[-1])))
        group_input_tokens_z = feature.gather(dim=1, index=torch.tile(center_z, (
            1, 1, feature.shape[-1])))
        center_xx = xyz.gather(dim=1, index=torch.tile(center_x, (1, 1, xyz.shape[-1])))
        center_yy = xyz.gather(dim=1, index=torch.tile(center_y, (1, 1, xyz.shape[-1])))
        center_zz = xyz.gather(dim=1, index=torch.tile(center_z, (1, 1, xyz.shape[-1])))
        group_input_tokens = torch.cat([group_input_tokens_x, group_input_tokens_y, group_input_tokens_z], dim=1)
        center = torch.cat([center_xx, center_yy, center_zz], dim=1)
        x = group_input_tokens

        # transformer
        x = blocks(x)
        return x, center.contiguous()

    def forward(self, xyz1, xyz2, points1, points2, xyz1_, xyz2_, points1_, points2_):
        """
        Cost Volume layer for Flow Estimation
        Input:
            xyz1: input points position data, [B, C, N1]
            xyz2: input points position data, [B, C, N2]
            points1: input points data, [B, D, N1]
            points2: input points data, [B, D, N2]
        Return:
            new_points: upsample points feature data, [B, D', N1]
        """
        # import ipdb; ipdb.set_trace()
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, _, N1_ = xyz1_.shape
        _, _, N2_ = xyz2_.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        _, D1_, _ = points1_.shape
        _, D2_, _ = points2_.shape
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        xyz1_ = xyz1_.permute(0, 2, 1)
        xyz2_ = xyz2_.permute(0, 2, 1)
        points1 = points1.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)
        points1_ = points1_.permute(0, 2, 1)
        points2_ = points2_.permute(0, 2, 1)

        # keypoint-to-keypoint Volume
        knn_idx = knn_point(self.nsample, xyz2_, xyz1_) # B, N1, nsample
        neighbor_xyz = index_points_group(xyz2_, knn_idx)
        direction_xyz = neighbor_xyz - xyz1_.view(B, N1_, 1, C)

        grouped_points2 = index_points_group(points2_, knn_idx) # B, N1, nsample, D2
        grouped_points1 = points1_.view(B, N1_, 1, D1_)
        # new_points = torch.cat([grouped_points1, grouped_points2, direction_xyz], dim = -1) # B, N1, nsample, D1+D2+3

        grouped_points2 = torch.cat((grouped_points2, neighbor_xyz), dim=-1)
        grouped_points1 = torch.cat((grouped_points1, xyz1_.view(B, N1_, 1, C)), dim=-1)

        std = torch.std((grouped_points2 - grouped_points1).reshape(B, -1), dim=-1, keepdim=True).unsqueeze(dim=-1).unsqueeze(dim=-1)
        grouped_points = (grouped_points2 - grouped_points1) / (std + 1e-5)
        grouped_points = self.affine_alpha_ * grouped_points + self.affine_beta_
        new_points = torch.cat([grouped_points1.repeat(1, 1, self.nsample, 1), grouped_points2, grouped_points], dim=-1)
        new_points = new_points.permute(0, 3, 2, 1) # [B, D1+D2+3, nsample, N1]
        for i, conv in enumerate(self.mlp_convs0):
            if self.bn:
                bn = self.mlp_bns0[i]
                new_points =  self.relu(bn(conv(new_points)))
            else:
                new_points =  self.relu(conv(new_points))

        a_norm = F.normalize(grouped_points1[:, :, :, :256], dim=-1)
        b_norm = F.normalize(grouped_points2[:, :, :, :256], dim=-1)
        similarity_matrix = torch.matmul(b_norm, a_norm.transpose(-1, -2))

        # weighted sum
        weights = self.weightnet0(direction_xyz.permute(0, 3, 2, 1)) # B C nsample N1
        weights_feat = self.weightnet0_feat(similarity_matrix.permute(0, 3, 2, 1)) # B C nsample N1

        keypoint_to_keypoint_cost = self.alph * torch.sum(weights * new_points, dim = 2) + (1 - self.alph) * torch.sum(weights_feat * new_points, dim = 2) # B C N

        # point-to-patch Volume
        knn_idx = knn_point(self.nsample, xyz2, xyz1) # B, N1, nsample
        neighbor_xyz = index_points_group(xyz2, knn_idx)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)

        grouped_points2 = index_points_group(points2, knn_idx) # B, N1, nsample, D2
        grouped_points1 = points1.view(B, N1, 1, D1)
        # new_points = torch.cat([grouped_points1, grouped_points2, direction_xyz], dim = -1) # B, N1, nsample, D1+D2+3

        grouped_points2 = torch.cat((grouped_points2, neighbor_xyz), dim=-1)
        grouped_points1 = torch.cat((grouped_points1, xyz1.view(B, N1, 1, C)), dim=-1)

        std = torch.std((grouped_points2 - grouped_points1).reshape(B, -1), dim=-1, keepdim=True).unsqueeze(dim=-1).unsqueeze(dim=-1)
        grouped_points = (grouped_points2 - grouped_points1) / (std + 1e-5)
        grouped_points = self.affine_alpha * grouped_points + self.affine_beta
        new_points = torch.cat([grouped_points1.repeat(1, 1, self.nsample, 1), grouped_points2, grouped_points], dim=-1)
        new_points = new_points.permute(0, 3, 2, 1)  # [B, D1+D2+3, nsample, N1]
        for i, conv in enumerate(self.mlp_convs):
            if self.bn:
                bn = self.mlp_bns[i]
                new_points =  self.relu(bn(conv(new_points)))
            else:
                new_points =  self.relu(conv(new_points))

        a_norm = F.normalize(grouped_points1[:, :, :, :256], dim=-1)
        b_norm = F.normalize(grouped_points2[:, :, :, :256], dim=-1)
        similarity_matrix = torch.matmul(b_norm, a_norm.transpose(-1, -2))

        # weighted sum
        weights = self.weightnet1(direction_xyz.permute(0, 3, 2, 1)) # B C nsample N1
        weights_feat = self.weightnet1_feat(similarity_matrix.permute(0, 3, 2, 1)) # B C nsample N1

        point_to_patch_cost = self.alph * torch.sum(weights * new_points, dim = 2) + (1 - self.alph) * torch.sum(weights_feat * new_points, dim = 2) # B C N

        # # Patch to Patch Cost
        knn_idx = knn_point(self.nsample, xyz1, xyz1)  # B, N1, nsample
        # knn_idx = pointutils.ball_query(4.0, self.nsample//2, xyz1.contiguous(), xyz1.contiguous())
        neighbor_xyz = index_points_group(xyz1, knn_idx).reshape(-1, self.nsample, C)
        neighbor_feat = index_points_group(point_to_patch_cost.permute(0, 2, 1), knn_idx).reshape(-1, self.nsample, D1)

        new_point_to_patch_cost, new_xyz1 = self.mamba(neighbor_xyz, neighbor_feat, self.blocks)
        patch_to_patch_cost = self.propagation(xyz1.reshape(-1, 1, C), new_xyz1,
                                               point_to_patch_cost.permute(0, 2, 1).reshape(-1, 1, D1), new_point_to_patch_cost)
        patch_to_patch_cost = patch_to_patch_cost.reshape(B, N1, D1).permute(0, 2, 1)

        # # keyPatch to keyPatch Cost
        knn_idx = knn_point(self.nsample, xyz1_, xyz1_)  # B, N1, nsample
        # knn_idx = pointutils.ball_query(8.0, self.nsample//2, xyz1_.contiguous(), xyz1_.contiguous())
        neighbor_xyz = index_points_group(xyz1_, knn_idx).reshape(-1, self.nsample, C)
        neighbor_feat = index_points_group(keypoint_to_keypoint_cost.permute(0, 2, 1), knn_idx).reshape(-1, self.nsample, D1)

        new_keypoint_to_keypoint_cost, new_xyz1_ = self.mamba(neighbor_xyz, neighbor_feat, self.blocks_)
        keypatch_to_keypatch_cost = self.propagation_(xyz1_.reshape(-1, 1, C), new_xyz1_, keypoint_to_keypoint_cost.permute(0, 2, 1).reshape(-1, 1, D1), new_keypoint_to_keypoint_cost)
        keypatch_to_keypatch_cost = keypatch_to_keypatch_cost.reshape(B, N1_, D1).permute(0, 2, 1)

        patch_to_patch_cost = torch.cat((patch_to_patch_cost, keypatch_to_keypatch_cost), dim=-1)

        return patch_to_patch_cost

