"""
Author: yida
Time is: 2021/11/13 10:21
this Code: 实现MobileNetV3-Small
"""
import os

import torch
import torch.nn as nn

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class MobileNetV3(nn.Module):
    def __init__(self, k):
        super(MobileNetV3, self).__init__()
        # 第一层
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, stride=2),
            nn.BatchNorm2d(16),
            nn.Hardswish(),
        )

        # 中间层, 使用bneck块
        self.layer2 = Bneck(input_size=16, operator_kernel=3, exp_size=16, out_size=16, NL='RE', s=2,
                            SE=True, skip_connection=False)

        self.layer3 = Bneck(input_size=32, operator_kernel=3, exp_size=72, out_size=24, NL='RE', s=2,
                            SE=False, skip_connection=False)

        self.layer4 = Bneck(input_size=64, operator_kernel=3, exp_size=88, out_size=24, NL='RE', s=1,
                            SE=False, skip_connection=True)

        self.layer5 = Bneck(input_size=64, operator_kernel=5, exp_size=96, out_size=40, NL='HS', s=2,
                            SE=True, skip_connection=False)

        self.layer6 = Bneck(input_size=128, operator_kernel=5, exp_size=240, out_size=40, NL='HS', s=1,
                            SE=True, skip_connection=True)

        self.layer7 = Bneck(input_size=128, operator_kernel=5, exp_size=240, out_size=40, NL='HS', s=1,
                            SE=True, skip_connection=True)

        self.layer8 = Bneck(input_size=256, operator_kernel=5, exp_size=120, out_size=48, NL='HS', s=1,
                            SE=True, skip_connection=False)

        self.layer9 = Bneck(input_size=256, operator_kernel=5, exp_size=144, out_size=48, NL='HS', s=1,
                            SE=True, skip_connection=True)

        self.layer10 = Bneck(input_size=512, operator_kernel=5, exp_size=288, out_size=96, NL='HS', s=2,
                             SE=True, skip_connection=False)

        self.layer11 = Bneck(input_size=512, operator_kernel=5, exp_size=576, out_size=96, NL='HS', s=1,
                             SE=True, skip_connection=False)

        self.layer12 = Bneck(input_size=512, operator_kernel=5, exp_size=576, out_size=96, NL='HS', s=1,
                             SE=True, skip_connection=True)

        # 结尾层
        self.layer13 = nn.Sequential(
            nn.Conv2d(96, 576, 1, stride=1),
            nn.BatchNorm2d(576),
            nn.Hardswish(),
        )

        # 池化
        self.layer14_pool = nn.AvgPool2d((7, 7), stride=1)

        # 使用1×1卷积替代全连接层
        self.layer15 = nn.Sequential(
            nn.Conv2d(576, 1024, 1, stride=1),
            nn.Hardswish(),  # 未使用BN层
        )

        self.layer16 = nn.Sequential(
            nn.Conv2d(1024, k, 1, stride=1),  # 未使用BN层
        )

    def forward(self, x):
        x = self.layer1(x)  # 第一层, s=2, 使用h-swish激活函数
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        x = self.layer12(x)
        print(x.shape)
        x = self.layer13(x)
        x = self.layer14_pool(x)
        x = self.layer15(x)
        x = self.layer16(x)
        x = torch.squeeze(x)  # 把[b, k, 1, 1] -> 压缩成[b, k]
        return x


class Bneck(nn.Module):
    def __init__(self, input_size, operator_kernel, exp_size, out_size, NL, s, SE=False,
                 skip_connection=False):
        """
        MobileNetV3的block块
        :param input_size: 输入维度
        :param operator_kernel: Dw卷积核大小
        :param exp_size: 升维维数
        :param out_size: 输出维数
        :param NL: 非线性激活函数,包含Relu以及h-switch
        :param s: 卷积核步矩
        :param SE: 是否使用注意力机制,默认为false
        :param skip_connection: 是否进行跳跃连接,当且仅当输入与输出维数相同且大小相同时开启
        """
        super(Bneck, self).__init__()
        # 1.使用1×1卷积升维
        self.conv_1_1_up = nn.Conv2d(input_size, exp_size, 1)
        if NL == 'RE':
            self.nl1 = nn.Sequential(
                nn.ReLU(),
                nn.BatchNorm2d(exp_size),
            )
        elif NL == 'HS':
            self.nl1 = nn.Sequential(
                nn.Hardswish(),
                nn.BatchNorm2d(exp_size),
            )
        # 2.使用Dwise卷积, groups与输入输出维度相同
        self.depth_conv = nn.Conv2d(exp_size, exp_size, kernel_size=operator_kernel, stride=s, groups=exp_size,
                                    padding=(operator_kernel - 1) // 2)  # 进行padding补零使shape减半时保存一致
        self.nl2 = nn.Sequential(
            self.nl1,
            nn.BatchNorm2d(exp_size)
        )

        #  3.使用1×1卷积降维
        self.conv_1_1_down = nn.Conv2d(exp_size, out_size, 1)

        # 判断是否添加注意力机制
        self.se = SE
        if SE:
            self.se_block = SEblock(exp_size)

        # 判断是否使用跳跃连接: 说明-> 当输入维数等于输出维数且大小相同时才进行跳跃连接
        self.skip = skip_connection

    def forward(self, x):
        # 1.1×1卷积升维
        x1 = self.conv_1_1_up(x)
        x1 = self.nl1(x1)

        # 2.Dwise卷积
        x2 = self.depth_conv(x1)
        x2 = self.nl2(x2)

        # 判断是否添加注意力机制
        if self.se:
            x2 = self.se_block(x2)

        # 3.1×1卷积降维
        x3 = self.conv_1_1_down(x2)

        # 判断是否使用跳跃连接
        if self.skip:
            x3 = x3 + x
        print("bneck:", x3.shape)
        return x3


class SEblock(nn.Module):
    def __init__(self, channel, r=0.25):
        """
        注意力机制模块
        :param channel: channel为输入的维度,
        :param r: r为全连接层缩放比例->控制中间层个数 默认为1/4
        """
        super(SEblock, self).__init__()
        # 全局均值池化
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(channel, int(channel * r)),
            nn.ReLU(),
            nn.Linear(int(channel * r), channel),
            nn.Sigmoid(),  # 原文中的hard-alpha 我不知道是什么激活函数,就用SE原文的Sigmoid替代(如果你知道是什么就把这儿的激活函数替换掉)
        )

    def forward(self, x):
        # 对x进行分支计算权重, 进行全局均值池化
        branch = self.global_avg_pool(x)
        branch = branch.view(branch.size(0), -1)

        # 全连接层得到权重
        weight = self.fc(branch)

        # 将维度为b, c的weight, reshape成b, c, 1, 1 与 输入x 相乘
        h, w = weight.shape
        weight = torch.reshape(weight, (h, w, 1, 1))

        # 乘积获得结果
        scale = weight * x
        return scale


if __name__ == '__main__':
    inputs = torch.randn(2, 3, 512, 512)
    model = MobileNetV3(k=1000
                        )  # k为分类数
    #print(model)

    outputs = model(inputs)


