
import torch.nn.functional as F
import torch
import torch.nn as nn

def calDist( fts, prototype, scaler=20):
    """
    Calculate the distance between features and prototypes

    Args:
        fts: input features
            expect shape: N x C x H x W
        prototype: prototype of one semantic class
            expect shape: base+Scribble+Scribble x C
    """
    # 余弦相似度计算，改为复合相似度计算
    dist = F.cosine_similarity(fts, prototype[..., None, None], dim=1) * scaler
    return dist

fts=torch.rand(1,3,2,2)*10
#print(fts)
prototype=torch.rand(1,3)
#print(prototype)
dis=F.cosine_similarity(fts, prototype[..., None, None], dim=1)
dis2=1/F.pairwise_distance(fts, prototype[..., None, None])
print(dis)
print(dis2)