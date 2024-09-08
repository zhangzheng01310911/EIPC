from __future__ import division
import os, math, datetime, time, json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import sys
sys.path.append("..")
from Models import model, basic
from Utils import util, dataset
from collections import OrderedDict
import numpy as np


class Tester:
    def __init__(self, config_dict):
        torch.manual_seed(config_dict['trainer']['seed'])
        torch.cuda.manual_seed(config_dict['trainer']['seed'])
        self.name = config_dict['name']
        self.namesave = config_dict['namesave']
        self.with_cuda = config_dict['tester']['with_cuda']
        self.batch_size = config_dict['tester']['batch_size']       
        print('**************** %s (Evaluation)****************' % (self.name))
        #! create folder to save results
        self.work_dir = os.path.join(config_dict['save_dir'], self.name)
        self.work_dir_save = os.path.join(config_dict['save_dir'], self.namesave)
        util.exists_or_mkdir(self.work_dir)
        util.exists_or_mkdir(self.work_dir_save)
        print("- working directory: %s"%self.work_dir)
        experiment_name = config_dict['dataset']['test'].split('/')[-2]

        self.cache_dir = os.path.join(self.work_dir_save, 'cache')
        self.val_dir = os.path.join(self.cache_dir, 'validation')
        util.exists_or_mkdir(self.val_dir, need_remove=False)
        self.val_encode_dir = os.path.join(self.val_dir, 'encode')
        self.val_decode_dir = os.path.join(self.val_dir, 'decode')

        self.result_dir = os.path.join(self.work_dir_save, experiment_name)
        util.exists_or_mkdir(self.result_dir, need_remove=False)
        self.encode_dir = os.path.join(self.result_dir, 'encode')
        self.encode_dir_mask = os.path.join(self.result_dir, 'encode_mask')
        self.decode_dir = os.path.join(self.result_dir, 'decode')
        util.exists_or_mkdir(self.encode_dir, need_remove=True)
        util.exists_or_mkdir(self.encode_dir_mask, need_remove=True)
        util.exists_or_mkdir(self.decode_dir, need_remove=True)
        #! evaluation dataset
        self.test_loader = dataset.create_dataloader(config_dict['dataset']['name'], config_dict['dataset']['test'],\
            self.batch_size, need_shuffle=False, is_mpdist=False)
        self.test_filenames = util.collect_filenames(config_dict['dataset']['test'])
        #! model definition
        guided_checkpt = config_dict['guide_model_path']
        self.colorization = model.ColorizationModel(guided_checkpt)
        self.model = eval('model.'+config_dict['model'])(inChannel=3, outChannel=1)
        if self.with_cuda:
            self.model = torch.nn.DataParallel(self.model).cuda()


    def _test(self, best_model=False):
        if best_model:
            print('@@@On best model.')
        #! loading pretrained model
        model_path = os.path.join(self.work_dir, 'model_last.pth.tar')
        if best_model:
            model_path = os.path.join(self.work_dir, 'model_best.pth.tar')
        if self._load_trainedModel(model_path) is False:
            return
        #! setting model mode
        self.model.eval()

        start_time = time.time()
        with torch.no_grad():
            avg_all_hints, avg_val_hints, cnt = 0, 0, 0
            for batch_idx, sample_batch in enumerate(self.test_loader):
                tm = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
                print('%s evaluating: [%d - %d]' % (tm, batch_idx*self.batch_size, (batch_idx+1)*self.batch_size))
                #! depatch sample list
                input_grays, target_ABs = sample_batch['grays'], sample_batch['ABs']
                input_colors = sample_batch['colors']
                #! transfer data to target device
                input_grays = input_grays.cuda(non_blocking=True)
                target_ABs = target_ABs.cuda(non_blocking=True)
                input_colors = input_colors.cuda(non_blocking=True)
                #! forward process

                InvLt = self.model(input_colors)
                InvL = torch.round(InvLt * 127.5 + 127.5) / 127.5 - 1.0
                InvLA=InvL

                # _, prob_maps, gate_maps = self.model(input_colors)
                # gate_maps = basic.dilate_seeds(gate_maps, kernel_size=5)
                color_seeds = target_ABs
                hint_masks = torch.ones_like(InvL)
                pred_ABs = self.colorization(InvLA, color_seeds, hint_masks)

                sparse_loss = 0
                pred_color = torch.cat((target_ABs, input_grays), dim=1)
                # data = {'pred_ABs':pred_ABs, 'target_ABs':target_ABs, 'sparse_loss':0, 'pred_gray':InvL}
                data = {'target_color': input_colors, 'pred_color': pred_color, 'sparse_loss': 0, 'pred_gray': InvL}
                #totalLoss_idx = self.trainLoss(data, epoch)
                #totalLoss_idx.backward()
                #self.optimizer.step()

                rgb_tensor = basic.lab2rgb(torch.cat((input_grays, pred_ABs), 1))
                rgb_imgs = basic.tensor2array(rgb_tensor * 2.0 - 1.0)
                gray_imgs = basic.tensor2array(InvL)
                # hint_masks = basic.tensor2array(gate_maps*2.0 - 1.0)
                # dilated_hints = basic.mark_color_hints(input_grays, target_ABs, gate_maps, kernel_size=3)
                # hint_masks = basic.tensor2array(dilated_hints*2.0 - 1.0)

                util.save_images_from_batch(gray_imgs, self.val_encode_dir, None, batch_idx)
                util.save_images_from_batch(rgb_imgs, self.val_decode_dir, None, batch_idx)

    def _load_trainedModel(self, model_path):
        if os.path.isfile(model_path) is False:
            print("@@@Warning:", model_path, " is invalid model location & exit ...")
            return False
        device = torch.device('cuda') if self.with_cuda is True else torch.device("cpu")
        checkpoint = torch.load(model_path, map_location=device)
        model_dict = checkpoint['model']
        if self.with_cuda:
            self.model.load_state_dict(model_dict, strict=True)
        else:
            new_model_dict = OrderedDict()
            for k, v in model_dict.items():
                name = k[7:]  # remove 7 chars 'module.'
                new_model_dict[name] = v
            self.model.load_state_dict(new_model_dict, strict=True)
        print("[*] pretrained model loaded successfully.")            
        return True


    def _colorize(self):
        start_time = time.time()
        with torch.no_grad():
            avg_psnr, cnt = 0, 0
            for batch_idx, sample_batch in enumerate(self.test_loader):
                tm = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
                print('%s evaluating: [%d - %d]' % (tm, batch_idx*self.batch_size, (batch_idx+1)*self.batch_size))
                #! depatch sample list
                input_grays, target_ABs = sample_batch['grays'], sample_batch['ABs']
                #! transfer data to target device
                input_grays = input_grays.cuda(non_blocking=True)
                target_ABs = target_ABs.cuda(non_blocking=True)
                #! forward process
                pred_ABs = self.colorization.testRandomHints(input_grays, target_ABs, point_num=50)
                name_list = self.test_filenames[batch_idx*self.batch_size:(batch_idx+1)*self.batch_size]
                #! save color images
                rgb_tensor = basic.lab2rgb(torch.cat((input_grays,pred_ABs), 1))
                rgb_imgs = basic.tensor2array(rgb_tensor*2.0 - 1.0)
                util.save_images_from_batch(rgb_imgs, self.result_dir, name_list, -1)
                if batch_idx >= 300:
                    break
        print("Testing finished! consumed %f sec" % (time.time() - start_time))


if __name__ == '__main__':
    print("FLAG: %s" % datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./PredictiveCoding_script.json', help='path of configuration file')
    parser.add_argument('--best', action='store_true', help='use the best or last model')
    args = parser.parse_args()
    if args.config_path is not None:
        config_dict = json.load(open(args.config_path))
        node = Tester(config_dict)
        #node._colorize()
        node._test(best_model=args.best)
    else:
        raise Exception("Unknow --config_path")
