"""
Fewshot Semantic Segmentation
"""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vgg import Encoder  #加载特征提取网络
from .vggg import Encoder2

from .SELayer import SELayer
from .CBAM import CBAM
from .CBAM_CA import CBAM_CA
from .CA import CoordAtt




class FewShotSeg(nn.Module):
    """
    Fewshot Segmentation model

    Args:
        in_channels:
            number of input channels
        pretrained_path:
            path of the model for initialization
        cfg:
            model configurations
    """
    def __init__(self, in_channels=3, pretrained_path=None, cfg=None):
        super().__init__()
        self.pretrained_path = pretrained_path
        self.config = cfg or {'align': False}

        # Encoder
        #self.encoder = nn.Sequential(OrderedDict([
        #    ('backbone', Encoder(in_channels, self.pretrained_path)),]))
        self.encoder = nn.Sequential(OrderedDict([
            ('backbone', Encoder(in_channels, self.pretrained_path)),]))
        self.encoder2 = nn.Sequential(OrderedDict([
            ('backbone2', Encoder2(in_channels, self.pretrained_path)),]))
        #self.SELayer = SELayer()
        #self.CBAM = CBAM(in_channel=512)
        #self.CBAM_CA = CBAM_CA()
        #self.CoordAtt = CoordAtt()
        #self.ASPP =ASPP()


    def forward(self, supp_imgs, fore_mask, back_mask, qry_imgs):
        """
        Args:
            supp_imgs: support images
                way x shot x [B x base+Scribble+Scribble x H x W], list of lists of tensors
            fore_mask: foreground masks for support images
                way x shot x [B x H x W], list of lists of tensors
            back_mask: background masks for support images
                way x shot x [B x H x W], list of lists of tensors
            qry_imgs: query images
                N x [B x base+Scribble+Scribble x H x W], list of tensors
        """
        n_ways = len(supp_imgs)
        n_shots = len(supp_imgs[0])
        n_queries = len(qry_imgs)
        batch_size = supp_imgs[0][0].shape[0]
        img_size = supp_imgs[0][0].shape[-2:]

        ###### Extract features ######
        imgs_concat = torch.cat([torch.cat(way, dim=0) for way in supp_imgs]
                                + [torch.cat(qry_imgs, dim=0),], dim=0)
        img_fts = self.encoder(imgs_concat)#原始
        img_fts2 = self.encoder2(imgs_concat)

        #img_fts = self.CBAM(img_fts)

        img_fts2 = torch.cat((img_fts,img_fts2),dim=1) #与中间层特征图融合
        #img_fts2 = self.SELayer(img_fts2)

        #img_fts2 = self.CBAM_CA(img_fts2)
        #img_fts2 = self.CoordAtt(img_fts2)


        fts_size = img_fts.shape[-2:]
        fts_size2 = img_fts2.shape[-2:]

        #获取support 和 query 特征图
        supp_fts = img_fts[:n_ways * n_shots * batch_size].view(
            n_ways, n_shots, batch_size, -1, *fts_size)  # Wa x Sh x B x C x H' x W'

        supp_fts2 = img_fts2[:n_ways * n_shots * batch_size].view(
            n_ways, n_shots, batch_size, -1, *fts_size)  # Wa x Sh x B x C x H' x W'

        qry_fts = img_fts[n_ways * n_shots * batch_size:].view(
            n_queries, batch_size, -1, *fts_size)   # N x B x C x H' x W'
        qry_fts2 = img_fts2[n_ways * n_shots * batch_size:].view(
            n_queries, batch_size, -1, *fts_size)   # N x B x C x H' x W'

        fore_mask = torch.stack([torch.stack(way, dim=0)
                                 for way in fore_mask], dim=0)  # Wa x Sh x B x H x W
        back_mask = torch.stack([torch.stack(way, dim=0)
                                 for way in back_mask], dim=0)  # Wa x Sh x B x H x W

        ###### Compute loss ######
        align_loss = 0
        outputs = []
        for epi in range(batch_size):
            ###### Extract prototype 从support的特征图中用最大池化提取######
            supp_fg_fts = [[self.getFeatures(supp_fts[way, shot, [epi]],
                                             fore_mask[way, shot, [epi]])
                            for shot in range(n_shots)] for way in range(n_ways)]

            supp_fg_fts2 = [[self.getFeatures(supp_fts2[way, shot, [epi]],
                                             fore_mask[way, shot, [epi]])
                            for shot in range(n_shots)] for way in range(n_ways)]

            supp_bg_fts = [[self.getFeatures(supp_fts[way, shot, [epi]],
                                             back_mask[way, shot, [epi]])
                            for shot in range(n_shots)] for way in range(n_ways)]

            supp_bg_fts2 = [[self.getFeatures(supp_fts2[way, shot, [epi]],
                                             fore_mask[way, shot, [epi]])
                            for shot in range(n_shots)] for way in range(n_ways)]

            ###### Obtain the prototypes######获得原型过程
            fg_prototypes, bg_prototype = self.getPrototype(supp_fg_fts, supp_bg_fts)
            fg_prototypes2, bg_prototype2 = self.getPrototype(supp_fg_fts2, supp_bg_fts2)

            ###### Compute the distance ######
            prototypes = [bg_prototype,] + fg_prototypes
            prototypes2 = [bg_prototype2, ] + fg_prototypes2

            dist = [self.calDist(qry_fts[:, epi], prototype) for prototype in prototypes]
            #dist2 = [self.calDist(qry_fts2[:, epi], prototype) for prototype in prototypes]
            #dist3 = [self.calDist(qry_fts[:, epi], prototype) for prototype in prototypes2]
            dist4 = [self.calDist(qry_fts2[:, epi], prototype) for prototype in prototypes2]


            pred = torch.stack(dist, dim=1)  # N x (base+Scribble+Scribble + Wa) x H' x W'
            #pred2 = torch.stack(dist2, dim=base+mutil_pro)  # N x (base+Scribble+Scribble + Wa) x H' x W'
            #pred3 = torch.stack(dist3, dim=base+mutil_pro)  # N x (base+Scribble+Scribble + Wa) x H' x W'
            pred4 = torch.stack(dist4, dim=1)  # N x (base+Scribble+Scribble + Wa) x H' x W'




            pred=(pred+pred4)/2

            outputs.append(F.interpolate(pred, size=img_size, mode='bilinear'))

            ###### Prototype alignment loss ######
            if self.config['align'] and self.training:
                align_loss_epi = self.alignLoss(qry_fts[:, epi], pred, supp_fts[:, :, epi],
                                                fore_mask[:, :, epi], back_mask[:, :, epi])
                align_loss += align_loss_epi

        output = torch.stack(outputs, dim=1)  # N x B x (base+Scribble+Scribble + Wa) x H x W
        output = output.view(-1, *output.shape[2:])
        return output, align_loss / batch_size


    def calDist(self, fts, prototype, scaler=20):
        """
        Calculate the distance between features and prototypes

        Args:
            fts: input features
                expect shape: N x C x H x W
            prototype: prototype of one semantic class
                expect shape: base+Scribble+Scribble x C
        """
        #余弦相似度计算，改为复合相似度计算
        #dist = F.cosine_similarity(fts, prototype[..., None, None], dim=base+hun55+mutil_pro) * scaler
        dist = (F.cosine_similarity(fts, prototype[..., None, None], dim=1) * 0.9 + (1/F.pairwise_distance(fts, prototype[
            ..., None, None])) * 0.1) * scaler
        #dist = F.cosine_similarity(fts, prototype[..., None, None], dim=base+hun91+mutil_pro+cbam) * scaler
        return dist


    def getFeatures(self, fts, mask):
        """
        Extract foreground and background features via masked average pooling

        Args:
            fts: input features, expect shape: base+Scribble+Scribble x C x H' x W'
            mask: binary mask, expect shape: base+Scribble+Scribble x H x W
        """
        fts = F.interpolate(fts, size=mask.shape[-2:], mode='bilinear')
        masked_fts = torch.sum(fts * mask[None, ...], dim=(2, 3)) \
            / (mask[None, ...].sum(dim=(2, 3)) + 1e-5) # base+Scribble+Scribble x C
        return masked_fts


    def getPrototype(self, fg_fts, bg_fts):
        """
        Average the features to obtain the prototype

        Args:
            fg_fts: lists of list of foreground features for each way/shot
                expect shape: Wa x Sh x [base+Scribble+Scribble x C]
            bg_fts: lists of list of background features for each way/shot
                expect shape: Wa x Sh x [base+Scribble+Scribble x C]
        """
        n_ways, n_shots = len(fg_fts), len(fg_fts[0])
        fg_prototypes = [sum(way) / n_shots for way in fg_fts]
        bg_prototype = sum([sum(way) / n_shots for way in bg_fts]) / n_ways
        return fg_prototypes, bg_prototype


    def alignLoss(self, qry_fts, pred, supp_fts, fore_mask, back_mask):
        """
        Compute the loss for the prototype alignment branch

        Args:
            qry_fts: embedding features for query images
                expect shape: N x C x H' x W'
            pred: predicted segmentation score
                expect shape: N x (base+Scribble+Scribble + Wa) x H x W
            supp_fts: embedding features for support images
                expect shape: Wa x Sh x C x H' x W'
            fore_mask: foreground masks for support images
                expect shape: way x shot x H x W
            back_mask: background masks for support images
                expect shape: way x shot x H x W
        """
        n_ways, n_shots = len(fore_mask), len(fore_mask[0])

        # Mask and get query prototype
        pred_mask = pred.argmax(dim=1, keepdim=True)  # N x base+Scribble+Scribble x H' x W'
        binary_masks = [pred_mask == i for i in range(1 + n_ways)]
        skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]
        pred_mask = torch.stack(binary_masks, dim=1).float()  # N x (base+Scribble+Scribble + Wa) x base+Scribble+Scribble x H' x W'
        qry_prototypes = torch.sum(qry_fts.unsqueeze(1) * pred_mask, dim=(0, 3, 4))
        qry_prototypes = qry_prototypes / (pred_mask.sum((0, 3, 4)) + 1e-5)  # (base+Scribble+Scribble + Wa) x C

        # Compute the support loss
        loss = 0
        for way in range(n_ways):
            if way in skip_ways:
                continue
            # Get the query prototypes
            prototypes = [qry_prototypes[[0]], qry_prototypes[[way + 1]]]
            for shot in range(n_shots):
                img_fts = supp_fts[way, [shot]]
                supp_dist = [self.calDist(img_fts, prototype) for prototype in prototypes]
                supp_pred = torch.stack(supp_dist, dim=1)
                supp_pred = F.interpolate(supp_pred, size=fore_mask.shape[-2:],
                                          mode='bilinear')
                # Construct the support Ground-Truth segmentation
                supp_label = torch.full_like(fore_mask[way, shot], 255,
                                             device=img_fts.device).long()
                supp_label[fore_mask[way, shot] == 1] = 1
                supp_label[back_mask[way, shot] == 1] = 0
                # Compute Loss
                loss = loss + F.cross_entropy(
                    supp_pred, supp_label[None, ...], ignore_index=255) / n_shots / n_ways
        return loss
