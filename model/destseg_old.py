import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import resnet18
from model.model_utils_old import ASPP, BasicBlock, l2_normalize, make_layer

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
import torchvision.transforms as transforms

class TeacherNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = timm.create_model(
            "resnet18",
            pretrained=True,
            features_only=True,
            out_indices=[1, 2, 3],
        )
        # freeze teacher model
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        self.eval()
        x1, x2, x3 = self.encoder(x)
        return (x1, x2, x3)


class StudentNet(nn.Module):
    def __init__(self, ed=True):
        super().__init__()
        self.ed = ed
        if self.ed:
            self.decoder_layer4 = make_layer(BasicBlock, 512, 512, 2)
            self.decoder_layer3 = make_layer(BasicBlock, 512, 256, 2)
            self.decoder_layer2 = make_layer(BasicBlock, 256, 128, 2)
            self.decoder_layer1 = make_layer(BasicBlock, 128, 64, 2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.encoder = timm.create_model(
            "resnet18",
            pretrained=False,
            features_only=True,
            out_indices=[1, 2, 3, 4],
        )

        _,self.bn = resnet18(pretrained=True)
    def forward(self, x):
        x1, x2, x3, x4 = self.encoder(x)
        # x = x4
        if not self.ed:
            return (x1, x2, x3)
        x = [x1,x2,x3]
        x = self.bn(x)
        b4 = self.decoder_layer4(x)
        b3 = F.interpolate(b4, size=x3.size()[2:], mode="bilinear", align_corners=False)
        b3 = self.decoder_layer3(b3)
        b2 = F.interpolate(b3, size=x2.size()[2:], mode="bilinear", align_corners=False)
        b2 = self.decoder_layer2(b2)
        b1 = F.interpolate(b2, size=x1.size()[2:], mode="bilinear", align_corners=False)
        b1 = self.decoder_layer1(b1)
        return (b1, b2, b3)


class SegmentationNet(nn.Module):
    def __init__(self, inplanes=448):
        super().__init__()
        self.res = make_layer(BasicBlock, inplanes, 256, 2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.head = nn.Sequential(
            ASPP(256, 256, [6, 12, 18]),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, 1),
        )

    def forward(self, x):
        x = self.res(x)
        x = self.head(x)
        x = torch.sigmoid(x)
        return x


class DeSTSeg(nn.Module):
    def __init__(self, dest=True, ed=True):
        super().__init__()
        self.teacher_net = TeacherNet()
        self.student_net = StudentNet(ed)
        self.dest = dest
        self.segmentation_net = SegmentationNet(inplanes=448)

    def forward(self, img_aug,img_origin=None):
        self.teacher_net.eval()

        if img_origin is None:  # for inference
            img_origin = img_aug.clone()

        outputs_teacher_aug = [
            l2_normalize(output_t.detach()) for output_t in self.teacher_net(img_aug)
        ]

        outputs_student_aug = [
            l2_normalize(output_s) for output_s in self.student_net(img_aug)
        ]
        # output_to_visualize = outputs_student_aug[0]
        # image_to_visualize = output_to_visualize[0, 0].detach().cpu().numpy()  # 形状 (64, 64)

        # # 插值到 256x256
        # image_to_visualize = F.interpolate(
        #     torch.tensor(image_to_visualize).unsqueeze(0).unsqueeze(0),  # 增加批次和通道维度
        #     size=(256, 256),
        #     mode='bilinear',
        #     align_corners=False
        # ).squeeze().numpy()  # 移除多余的维度

        # # 归一化到 0-255 并转换为 uint8
        # image_to_visualize = (image_to_visualize - np.min(image_to_visualize)) / (np.max(image_to_visualize) - np.min(image_to_visualize)) * 255
        # image_to_visualize = image_to_visualize.astype(np.uint8)

        # # 保存可视化
        # output_path = 'output_visualization.jpg'
        # cv2.imwrite(output_path, image_to_visualize)
        output = torch.cat(
            [
                F.interpolate(
                    -output_t * output_s,
                    size=outputs_student_aug[0].size()[2:],
                    mode="bilinear",
                    align_corners=False,
                )
                for output_t, output_s in zip(outputs_teacher_aug, outputs_student_aug)
            ],
            dim=1,
        )
        #         #         # 创建保存目录
        # save_dir = os.path.join('saved_new',category)
        # os.makedirs(save_dir, exist_ok=True)
        # to_pil = transforms.ToPILImage()
        # # 对通道维度进行平均
        # rgb_images = []  # 创建一个空列表以存储每个批次的 RGB 图像
        # feature_map = torch.mean(output, dim=1)
        # for i in range(feature_map.shape[0]):  # 遍历每个批次
        #     feature = feature_map[i].unsqueeze(0) # 形状变为 [64, 64]
        #     # feature_upsampled = torch.nn.functional.interpolate(feature.unsqueeze(0), size=(256, 256), mode='nearest').squeeze(0)
        #     # 归一化到 [0, 1]
        #     feature_upsampled = (feature - feature.min()) / (feature.max() - feature.min() + 1e-8)  
            
        #     # 转换为 PIL 图像
        #     rgb_image = to_pil(feature_upsampled)  
        #     rgb_image.save(os.path.join(save_dir, f'de_st_{fill_name[i]}.png'))  # 保存为 PNG 文件
        #     rgb_images.append(rgb_image)
        # print(f"Saved {feature_map.shape[0]} images to '{save_dir}' directory.")

        # output_test = torch.cat(
        #     [
        #         F.interpolate(
        #             output_s,
        #             size=outputs_student_aug[0].size()[2:],  # 上采样到相同的空间大小
        #             mode="bilinear",
        #             align_corners=False,
        #         )
        #         for output_s in outputs_student_aug  # 只遍历学生模型的输出
        #     ],
        #     dim=1,  # 在通道维度上连接
        # )

        # # rgb_images = []  # 创建一个空列表以存储每个批次的 RGB 图像
        # # 对通道维度进行平均
        # feature_map = torch.mean(output_test, dim=1)
        # for i in range(feature_map.shape[0]):  # 遍历每个批次
        #     feature = feature_map[i].unsqueeze(0) # 形状变为 [64, 64]
        #     # feature_upsampled = torch.nn.functional.interpolate(feature.unsqueeze(0), size=(256, 256), mode='nearest').squeeze(0)
        #     # 归一化到 [0, 1]
        #     feature_upsampled = (feature - feature.min()) / (feature.max() - feature.min() + 1e-8)  
            
        #     # 转换为 PIL 图像
        #     rgb_image = to_pil(feature_upsampled)  
        #     # rgb_images.append(rgb_image)
        #     rgb_image.save(os.path.join(save_dir, f'recon_{fill_name[i]}.png'))  # 保存为 PNG 文件

        # print(f"Saved {feature_map.shape[0]} images to '{save_dir}' directory.")
        # output_to_visualize = outputs_student_aug[0]
        # image_to_visualize = output_to_visualize[0, 0].detach().cpu().numpy()  # 形状 (64, 64)

        # # 插值到 256x256
        # image_to_visualize = F.interpolate(
        #     torch.tensor(image_to_visualize).unsqueeze(0).unsqueeze(0),  # 增加批次和通道维度
        #     size=(256, 256),
        #     mode='bilinear',
        #     align_corners=False
        # ).squeeze().numpy()  # 移除多余的维度

        # # 归一化到 0-255 并转换为 uint8
        # image_to_visualize = (image_to_visualize - np.min(image_to_visualize)) / (np.max(image_to_visualize) - np.min(image_to_visualize)) * 255
        # image_to_visualize = image_to_visualize.astype(np.uint8)

        # # 保存可视化
        # output_path = 'output_visualization.jpg'
        # cv2.imwrite(output_path, image_to_visualize)

        output_segmentation = self.segmentation_net(output)

        if self.dest:
            outputs_student = outputs_student_aug
        else:
            outputs_student = [
                l2_normalize(output_s) for output_s in self.student_net(img_origin)
            ]
        outputs_teacher = [
            l2_normalize(output_t.detach()) for output_t in self.teacher_net(img_origin)
        ]
        output_de_st_list = []
        for output_t, output_s in zip(outputs_teacher, outputs_student):
            a_map = 1 - torch.sum(output_s * output_t, dim=1, keepdim=True)
            output_de_st_list.append(a_map)
        output_de_st = torch.cat(
            [
                F.interpolate(
                    output_de_st_instance,
                    size=outputs_student[0].size()[2:],
                    mode="bilinear",
                    align_corners=False,
                )
                for output_de_st_instance in output_de_st_list
            ],
            dim=1,
        )  # [N, 3, H, W]


        output_de_st = torch.prod(output_de_st, dim=1, keepdim=True)
        contrast = [outputs_teacher_aug,outputs_student_aug]
        # return output_segmentation, output_de_st, output_de_st_list,contrast,rgb_images
        return output_segmentation, output_de_st, output_de_st_list,contrast
