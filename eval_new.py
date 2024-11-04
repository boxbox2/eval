import argparse
import os
import shutil
import warnings
from tqdm import tqdm
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchmetrics import AUROC, AveragePrecision

from constant import RESIZE_SHAPE, NORMALIZE_MEAN, NORMALIZE_STD, ALL_CATEGORY
#old
from data.mvtec_dataset import MVTecDataset

# ##new 
# from data.mvtec_dataset_np import MVTecDataset
from model.destseg_old import DeSTSeg
# from model.destseg_old_copy import DeSTSeg_old
from model.metrics import AUPRO, IAPS
import matplotlib.pyplot as plt
import numpy as np
import csv
warnings.filterwarnings("ignore")


def evaluate(args, category, model, visualizer, global_step=0):
    model.eval()
    with torch.no_grad():
        #old
        dataset = MVTecDataset(
            is_train=False,
            mvtec_dir=args.mvtec_path + category + "/test/",
            resize_shape=RESIZE_SHAPE,
            normalize_mean=NORMALIZE_MEAN,
            normalize_std=NORMALIZE_STD,
        )


        dataloader = DataLoader(
            dataset, batch_size=args.bs, shuffle=False, num_workers=args.num_workers
        )
        de_st_IAPS = IAPS().cuda()
        de_st_AUPRO = AUPRO().cuda()
        de_st_AUROC = AUROC().cuda()
        de_st_AP = AveragePrecision().cuda()
        de_st_detect_AUROC = AUROC().cuda()
        seg_IAPS = IAPS().cuda()
        seg_AUPRO = AUPRO().cuda()
        seg_AUROC = AUROC().cuda()
        seg_AP = AveragePrecision().cuda()
        seg_detect_AUROC = AUROC().cuda()

        for _, sample_batched in enumerate(tqdm(dataloader, desc="Evaluating")):
            img = sample_batched["img"].cuda()
            mask = sample_batched["mask"].to(torch.int64).cuda()
            filenames = sample_batched["file"]
            output_segmentation, output_de_st, output_de_st_list,contrast = model(img)
            # output_segmentation, output_de_st, output_de_st_list = model(img)
            output_segmentation = F.interpolate(
                output_segmentation,
                size=mask.size()[2:],
                mode="bilinear",
                align_corners=False,
            )
            output_de_st = F.interpolate(
                output_de_st, size=mask.size()[2:], mode="bilinear", align_corners=False
            )


            mask_sample = torch.max(mask.view(mask.size(0), -1), dim=1)[0]
            output_segmentation_sample, _ = torch.sort(
                output_segmentation.view(output_segmentation.size(0), -1),
                dim=1,
                descending=True,
            )
            output_segmentation_sample = torch.mean(
                output_segmentation_sample[:, : args.T], dim=1
            )
            output_de_st_sample, _ = torch.sort(
                output_de_st.view(output_de_st.size(0), -1), dim=1, descending=True
            )
            output_de_st_sample = torch.mean(output_de_st_sample[:, : args.T], dim=1)

            de_st_IAPS.update(output_de_st, mask)
            de_st_AUPRO.update(output_de_st, mask)
            de_st_AP.update(output_de_st.flatten(), mask.flatten())
            de_st_AUROC.update(output_de_st.flatten(), mask.flatten())
            de_st_detect_AUROC.update(output_de_st_sample, mask_sample)

            seg_IAPS.update(output_segmentation, mask)
            seg_AUPRO.update(output_segmentation, mask)
            seg_AP.update(output_segmentation.flatten(), mask.flatten())
            seg_AUROC.update(output_segmentation.flatten(), mask.flatten())
            seg_detect_AUROC.update(output_segmentation_sample, mask_sample)


            save_dir = os.path.join('./output',f"{args.datasource}", category)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
                        
        iap_de_st, iap90_de_st = de_st_IAPS.compute()
        aupro_de_st, ap_de_st, auc_de_st, auc_detect_de_st = (
            de_st_AUPRO.compute(),
            de_st_AP.compute(),
            de_st_AUROC.compute(),
            de_st_detect_AUROC.compute(),
        )
        iap_seg, iap90_seg = seg_IAPS.compute()
        aupro_seg, ap_seg, auc_seg, auc_detect_seg = (
            seg_AUPRO.compute(),
            seg_AP.compute(),
            seg_AUROC.compute(),
            seg_detect_AUROC.compute(),
        )

        visualizer.add_scalar("DeST_IAP", iap_de_st, global_step)
        visualizer.add_scalar("DeST_IAP90", iap90_de_st, global_step)
        visualizer.add_scalar("DeST_AUPRO", aupro_de_st, global_step)
        visualizer.add_scalar("DeST_AP", ap_de_st, global_step)
        visualizer.add_scalar("DeST_AUC", auc_de_st, global_step)
        visualizer.add_scalar("DeST_detect_AUC", auc_detect_de_st, global_step)

        visualizer.add_scalar("DeSTSeg_IAP", iap_seg, global_step)
        visualizer.add_scalar("DeSTSeg_IAP90", iap90_seg, global_step)
        visualizer.add_scalar("DeSTSeg_AUPRO", aupro_seg, global_step)
        visualizer.add_scalar("DeSTSeg_AP", ap_seg, global_step)
        visualizer.add_scalar("DeSTSeg_AUC", auc_seg, global_step)
        visualizer.add_scalar("DeSTSeg_detect_AUC", auc_detect_seg, global_step)


        print("Eval at step", global_step)
        print("================================")
        print("Denoising Student-Teacher (DeST)")
        print("pixel_AUC:", round(float(auc_de_st), 4))
        print("pixel_AP:", round(float(ap_de_st), 4))
        print("PRO:", round(float(aupro_de_st), 4))
        print("image_AUC:", round(float(auc_detect_de_st), 4))
        print("IAP:", round(float(iap_de_st), 4))
        print("IAP90:", round(float(iap90_de_st), 4))
        print()
        print("Segmentation Guided Denoising Student-Teacher (DeSTSeg)")
        print("pixel_AUC:", round(float(auc_seg), 4))
        print("pixel_AP:", round(float(ap_seg), 4))
        print("PRO:", round(float(aupro_seg), 4))
        print("image_AUC:", round(float(auc_detect_seg), 4))
        print("IAP:", round(float(iap_seg), 4))
        print("IAP90:", round(float(iap90_seg), 4))
        print()

        tmp_pap = round(float(ap_seg), 4)
        tmp_pauc = round(float(auc_seg), 4)
                #数据
        data = [
            ["Eval at step", global_step],
            
            # Denoising Student-Teacher (DeST)
            ["Denoising Student-Teacher (DeST)"],
            ["pixel_AUC", round(float(auc_de_st), 4)],
            ["pixel_AP", round(float(ap_de_st), 4)],
            ["PRO", round(float(aupro_de_st), 4)],
            ["image_AUC", round(float(auc_detect_de_st), 4)],
            ["IAP", round(float(iap_de_st), 4)],
            ["IAP90", round(float(iap90_de_st), 4)],
            
            # 空行用于区分两个块
            [""],
            
            # Segmentation Guided Denoising Student-Teacher (DeSTSeg)
            ["Segmentation Guided Denoising Student-Teacher (DeSTSeg)"],
            ["pixel_AUC", round(float(auc_seg), 4)],
            ["pixel_AP", round(float(ap_seg), 4)],
            ["PRO", round(float(aupro_seg), 4)],
            ["image_AUC", round(float(auc_detect_seg), 4)],
            ["IAP", round(float(iap_seg), 4)],
            ["IAP90", round(float(iap90_seg), 4)]
        ]


        file_path = os.path.join(save_dir, 'evaluation_results.csv')

        # 写入CSV文件
        with open(file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            for row in data:
                writer.writerow(row)


        de_st_IAPS.reset()
        de_st_AUPRO.reset()
        de_st_AUROC.reset()
        de_st_AP.reset()
        de_st_detect_AUROC.reset()
        seg_IAPS.reset()
        seg_AUPRO.reset()
        seg_AUROC.reset()
        seg_AP.reset()
        seg_detect_AUROC.reset()

        return tmp_pauc,tmp_pap


def test(args, category):
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    run_name = f"DeSTSeg_MVTec_test_{category}"
    if os.path.exists(os.path.join(args.log_path, run_name + "/")):
        shutil.rmtree(os.path.join(args.log_path, run_name + "/"))

    visualizer = SummaryWriter(log_dir=os.path.join(args.log_path, run_name + "/"))

    model = DeSTSeg(dest=True, ed=True).cuda()

    assert os.path.exists(
        os.path.join(args.checkpoint_path, args.base_model_name + category + "_best.pckl")
    )
    model.load_state_dict(
        torch.load(
            os.path.join(
                args.checkpoint_path, args.base_model_name + category + "_best.pckl"
            )
        )
    )

    evaluate(args, category, model, visualizer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--datasource", type=str, default="MVTEC")

    parser.add_argument("--mvtec_path", type=str, default="/data/MVTec-AD/")
    parser.add_argument("--test_path", type=str, default="/data/MVTec-AD/")
    parser.add_argument("--dtd_path", type=str, default="/data/dtd/images/")
    parser.add_argument("--checkpoint_path", type=str, default="./saved_model/")

    parser.add_argument("--base_model_name", type=str, default="DeSTSeg_MVTec_6000_")
    parser.add_argument("--log_path", type=str, default="./logs/")

    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--T", type=int, default=100)  # for image-level inference

    parser.add_argument("--category", nargs="*", type=str, default=ALL_CATEGORY)
    args = parser.parse_args()

    obj_list = args.category
    for obj in obj_list:
        assert obj in ALL_CATEGORY

    with torch.cuda.device(args.gpu_id):
        for obj in obj_list:
            print(obj)
            test(args, obj)