import sys
sys.path.append('/home/songjian/project/UrbanMamba')

import argparse
import os
import time

import numpy as np

from UrbanMamba.semanticsegmentation.configs.config import get_config

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from UrbanMamba.semanticsegmentation.datasets.make_data_loader import make_data_loader, MultimodalDamageAssessmentDatset
from UrbanMamba.semanticsegmentation.utils_func.metrics import Evaluator
from UrbanMamba.semanticsegmentation.models.ChangeMambaMMBDA import ChangeMambaMMBDA

import UrbanMamba.semanticsegmentation.utils_func.lovasz_loss as L
import torch.nn as nn

class Trainer(object):
    def __init__(self, args):
        self.args = args
        config = get_config(args)

        self.train_data_loader = make_data_loader(args)

        self.evaluator_loc = Evaluator(num_class=2)
        self.evaluator_clf = Evaluator(num_class=4)
        self.evaluator_total = Evaluator(num_class=4)
        self.evaluator_event_noto = Evaluator(num_class=4)
        self.evaluator_event_marshall = Evaluator(num_class=4)

        self.deep_model = ChangeMambaMMBDA(
            output_building=2, output_damage=4, 
            pretrained=args.pretrained_weight_path,
            patch_size=config.MODEL.VSSM.PATCH_SIZE, 
            in_chans=config.MODEL.VSSM.IN_CHANS, 
            num_classes=config.MODEL.NUM_CLASSES, 
            depths=config.MODEL.VSSM.DEPTHS, 
            dims=config.MODEL.VSSM.EMBED_DIM, 
            # ===================
            ssm_d_state=config.MODEL.VSSM.SSM_D_STATE,
            ssm_ratio=config.MODEL.VSSM.SSM_RATIO,
            ssm_rank_ratio=config.MODEL.VSSM.SSM_RANK_RATIO,
            ssm_dt_rank=("auto" if config.MODEL.VSSM.SSM_DT_RANK == "auto" else int(config.MODEL.VSSM.SSM_DT_RANK)),
            ssm_act_layer=config.MODEL.VSSM.SSM_ACT_LAYER,
            ssm_conv=config.MODEL.VSSM.SSM_CONV,
            ssm_conv_bias=config.MODEL.VSSM.SSM_CONV_BIAS,
            ssm_drop_rate=config.MODEL.VSSM.SSM_DROP_RATE,
            ssm_init=config.MODEL.VSSM.SSM_INIT,
            forward_type=config.MODEL.VSSM.SSM_FORWARDTYPE,
            # ===================
            mlp_ratio=config.MODEL.VSSM.MLP_RATIO,
            mlp_act_layer=config.MODEL.VSSM.MLP_ACT_LAYER,
            mlp_drop_rate=config.MODEL.VSSM.MLP_DROP_RATE,
            # ===================
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            patch_norm=config.MODEL.VSSM.PATCH_NORM,
            norm_layer=config.MODEL.VSSM.NORM_LAYER,
            downsample_version=config.MODEL.VSSM.DOWNSAMPLE,
            patchembed_version=config.MODEL.VSSM.PATCHEMBED,
            gmlp=config.MODEL.VSSM.GMLP,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
            ) 
       
        self.deep_model = self.deep_model.cuda()
        self.model_save_path = os.path.join(args.model_param_path, args.dataset,
                                            args.model_type + '_' + str(time.time()))
        self.lr = args.learning_rate

        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model_dict = {}
            state_dict = self.deep_model.state_dict()
            for k, v in checkpoint.items():
                if k in state_dict:
                    model_dict[k] = v
            state_dict.update(model_dict)
            self.deep_model.load_state_dict(state_dict)

        self.optim = optim.AdamW(self.deep_model.parameters(),
                                 lr=args.learning_rate,
                                 weight_decay=args.weight_decay)

    def training(self):
        best_mIoU = 0.0
        best_round = []
        torch.cuda.empty_cache()
        elem_num = len(self.train_data_loader)
        train_enumerator = enumerate(self.train_data_loader)
        class_weights = torch.FloatTensor([1, 1, 1, 1]).cuda()

        for _ in tqdm(range(elem_num)):
            itera, data = train_enumerator.__next__()
            pre_change_imgs, post_change_imgs, labels_loc, labels_clf, _ = data

            pre_change_imgs = pre_change_imgs.cuda()
            post_change_imgs = post_change_imgs.cuda()
            labels_loc = labels_loc.cuda().long()
            labels_clf = labels_clf.cuda().long()

            
            valid_labels_clf = (labels_clf != 255).any()
            if not valid_labels_clf:
               continue
            
            output_loc, output_clf = self.deep_model(pre_change_imgs, post_change_imgs)

            self.optim.zero_grad()

            ce_loss_loc = F.cross_entropy(output_loc, labels_loc, ignore_index=255)
            lovasz_loss_loc = L.lovasz_softmax(F.softmax(output_loc, dim=1), labels_loc, ignore=255)
            
            ce_loss_clf = F.cross_entropy(output_clf, labels_clf, weight = class_weights, ignore_index=255)
            lovasz_loss_clf = L.lovasz_softmax(F.softmax(output_clf, dim=1), labels_clf, ignore=255)
            final_loss = ce_loss_loc + ce_loss_clf + (0.5 * lovasz_loss_loc + 0.75 * lovasz_loss_clf)
            # final_loss = main_loss

            final_loss.backward()

            self.optim.step()

            if (itera + 1) % 10 == 0:
                print(f'iter is {itera + 1}, classification loss is {ce_loss_clf + lovasz_loss_clf}')
                if (itera + 1) % 500 == 0:
                    self.deep_model.eval()
                    loc_f1_score_val, harmonic_mean_f1_val, final_OA_val, mIoU_val, IoU_of_each_class_val = self.validation()
                    loc_f1_score_test, harmonic_mean_f1_test, final_OA_test, mIoU_test, IoU_of_each_class_test = self.test()

                    if mIoU_val > best_mIoU:
                        torch.save(self.deep_model.state_dict(), os.path.join(self.model_save_path, f'best_model.pth'))
                        best_mIoU = mIoU_val
                        best_round = {
                            'best iter': itera + 1,
                            'loc f1 (val)': loc_f1_score_val * 100,
                            'clf f1 (val)': harmonic_mean_f1_val * 100,
                            'OA (val)': final_OA_val * 100,
                            'mIoU (val)': mIoU_val * 100,
                            'sub class IoU (val)': IoU_of_each_class_val * 100,
                            'loc f1 (test)': loc_f1_score_test * 100,
                            'clf f1 (test)': harmonic_mean_f1_test * 100,
                            'OA (test)': final_OA_test * 100,
                            'mIoU (test)': mIoU_test * 100,
                            'sub class IoU (test)': IoU_of_each_class_test * 100
                        }
                    self.deep_model.train()

        print('The accuracy of the best round is ', best_round)


    def validation(self):
        print('---------starting validation-----------')
        self.evaluator_loc.reset()
        self.evaluator_clf.reset()
        self.evaluator_total.reset()
        dataset = MultimodalDamageAssessmentDatset(self.args.val_dataset_path, self.args.val_data_name_list, 256, None, 'test')
        val_data_loader = DataLoader(dataset, batch_size=4, num_workers=1, drop_last=False)
        torch.cuda.empty_cache()

        # vbar = tqdm(val_data_loader, ncols=50)
        with torch.no_grad():

            for itera, data in enumerate(val_data_loader):
                pre_change_imgs, post_change_imgs, labels_loc, labels_clf, _ = data

                pre_change_imgs = pre_change_imgs.cuda()
                post_change_imgs = post_change_imgs.cuda()
                labels_loc = labels_loc.cuda().long()
                labels_clf = labels_clf.cuda().long()


                # i#nput_data = torch.cat([pre_change_imgs, post_change_imgs], dim=1)
                # output_clf = self.deep_model(input_data)
                output_loc, output_clf = self.deep_model(pre_change_imgs, post_change_imgs)

                output_loc = output_loc.data.cpu().numpy()
                output_loc = np.argmax(output_loc, axis=1)
                labels_loc = labels_loc.cpu().numpy()

                output_clf = output_clf.data.cpu().numpy()
                output_clf = np.argmax(output_clf, axis=1)
                
                labels_clf = labels_clf.cpu().numpy()

                self.evaluator_loc.add_batch(labels_loc, output_loc)
                
                output_clf_damage_part = output_clf[labels_loc > 0]
                labels_clf_damage_part = labels_clf[labels_loc > 0]
                self.evaluator_clf.add_batch(labels_clf_damage_part, output_clf_damage_part)

                self.evaluator_total.add_batch(labels_clf, output_clf)

        loc_f1_score = self.evaluator_loc.Pixel_F1_score()
        damage_f1_score = self.evaluator_clf.Damage_F1_score()
        harmonic_mean_f1 = len(damage_f1_score) / np.sum(1.0 / damage_f1_score)
        final_OA = self.evaluator_total.Pixel_Accuracy()
        IoU_of_each_class = self.evaluator_total.Intersection_over_Union()
        mIoU = self.evaluator_total.Mean_Intersection_over_Union()
        print(f'OA is {100 * final_OA}, mIoU is {100 * mIoU}, sub class IoU is {100 * IoU_of_each_class}')
        return loc_f1_score, harmonic_mean_f1, final_OA, mIoU, IoU_of_each_class
    

    def test(self):
        print('---------starting testing-----------')
        self.evaluator_loc.reset()
        self.evaluator_clf.reset()
        self.evaluator_total.reset()
        dataset = MultimodalDamageAssessmentDatset(self.args.test_dataset_path, self.args.test_data_name_list, 256, None, 'test', suffix='.tif')
        val_data_loader = DataLoader(dataset, batch_size=4, num_workers=1, drop_last=False)
        torch.cuda.empty_cache()

        # vbar = tqdm(val_data_loader, ncols=50)
        with torch.no_grad():
            for itera, data in enumerate(val_data_loader):
                pre_change_imgs, post_change_imgs, labels_loc, labels_clf, _ = data

                pre_change_imgs = pre_change_imgs.cuda()
                post_change_imgs = post_change_imgs.cuda()
                labels_loc = labels_loc.cuda().long()
                labels_clf = labels_clf.cuda().long()


                output_loc, output_clf = self.deep_model(pre_change_imgs, post_change_imgs)

                output_loc = output_loc.data.cpu().numpy()
                output_loc = np.argmax(output_loc, axis=1)
                labels_loc = labels_loc.cpu().numpy()

                output_clf = output_clf.data.cpu().numpy()
                output_clf = np.argmax(output_clf, axis=1)
                
                labels_clf = labels_clf.cpu().numpy()

                self.evaluator_loc.add_batch(labels_loc, output_loc)
                
                output_clf_damage_part = output_clf[labels_loc > 0]
                labels_clf_damage_part = labels_clf[labels_loc > 0]
                self.evaluator_clf.add_batch(labels_clf_damage_part, output_clf_damage_part)

                self.evaluator_total.add_batch(labels_clf, output_clf)

        loc_f1_score = self.evaluator_loc.Pixel_F1_score()
        damage_f1_score = self.evaluator_clf.Damage_F1_score()
        harmonic_mean_f1 = len(damage_f1_score) / np.sum(1.0 / damage_f1_score)
        final_OA = self.evaluator_total.Pixel_Accuracy()
        IoU_of_each_class = self.evaluator_total.Intersection_over_Union()
        mIoU = self.evaluator_total.Mean_Intersection_over_Union()
        print(f'OA is {100 * final_OA}, mIoU is {100 * mIoU}, sub class IoU is {100 * IoU_of_each_class}')
        return loc_f1_score, harmonic_mean_f1, final_OA, mIoU, IoU_of_each_class



def main():
    parser = argparse.ArgumentParser(description="Training on xBD dataset")
    parser.add_argument('--cfg', type=str, default='/home/songjian/project/UrbanMamba/VMamba/classification/configs/vssm1/vssm_base_224.yaml')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('--pretrained_weight_path', type=str)

    parser.add_argument('--dataset', type=str, default='xBD')
    parser.add_argument('--type', type=str, default='train')
    parser.add_argument('--train_dataset_path', type=str)
    parser.add_argument('--test_dataset_path', type=str)
    parser.add_argument('--val_dataset_path', type=str)

    parser.add_argument('--train_data_list_path', type=str)
    parser.add_argument('--test_data_list_path', type=str)
    parser.add_argument('--val_data_list_path', type=str)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--train_batch_size', type=int, default=4)
    parser.add_argument('--crop_size', type=int, default=512)
    parser.add_argument('--train_data_name_list', type=list)
    parser.add_argument('--test_data_name_list', type=list)
    parser.add_argument('--val_data_name_list', type=list)

    parser.add_argument('--start_iter', type=int, default=0)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--max_iters', type=int, default=240000)
    parser.add_argument('--model_type', type=str, default='MMMambaBDA')
    parser.add_argument('--model_param_path', type=str, default='../saved_models')

    parser.add_argument('--resume', type=str)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-3)
    parser.add_argument('--num_workers', type=int)

    args = parser.parse_args()
    with open(args.train_data_list_path, "r") as f:
        # data_name_list = f.read()
        data_name_list = [data_name.strip() for data_name in f]
    args.train_data_name_list = data_name_list

    with open(args.test_data_list_path, "r") as f:
        # data_name_list = f.read()
        test_data_name_list = [data_name.strip() for data_name in f]
    args.test_data_name_list = test_data_name_list

    with open(args.val_data_list_path, "r") as f:
        # data_name_list = f.read()
        val_data_name_list = [data_name.strip() for data_name in f]
    args.val_data_name_list = val_data_name_list

    trainer = Trainer(args)
    trainer.training()


if __name__ == "__main__":
    main()
