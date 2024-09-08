from __future__ import division
import os, math, datetime, time, json
import pdb, numpy
import sys
sys.path.append("..")
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from Models import model, basic, loss
from Utils import util, dataset
from collections import OrderedDict

class Trainer:
    def __init__(self, config_dict, resume=False, mpdist=True, gpu_num=4, gpu_no=0):
        torch.manual_seed(config_dict['trainer']['seed'])
        torch.cuda.manual_seed(config_dict['trainer']['seed'])
        #! parsing training configuration
        self.name = config_dict['name']
        self.n_epochs = config_dict['trainer']['n_epochs']
        self.batch_size = config_dict['trainer']['batch_size']
        self.need_valid = config_dict['trainer']['need_valid']
        self.config_dict = config_dict
        self.monitorMetric = 9999
        self.start_epoch = 0
        self.resume_mode = resume
        self.mpdist = mpdist
        self.gpu_no = gpu_no
        
        '''set model, loss and optimization'''
        self.model = eval('model.'+config_dict['model'])(inChannel=3, outChannel=1)
        param_count = basic.getParamsAmount(self.model)
        if self.mpdist:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model.cuda(gpu_no), device_ids=[gpu_no])
        else:
            self.model = torch.nn.DataParallel(self.model).cuda()
        guided_checkpt = config_dict['guide_model_path']
        self.colorization = model.ColorizationModel(guided_checkpt, self.mpdist, self.gpu_no)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config_dict['trainer']['lr'])
        self.learningrateList = []
        self.work_dir = os.path.join(config_dict['save_dir'], self.name)
        self.cache_dir = os.path.join(self.work_dir, 'cache')     
        if self.resume_mode:
            self._resume_checkpoint()
        
        '''learning rate scheduler'''
        #decay_ratio = 1.0/300
        #decay_epochs = self.n_epochs
        #polynomial_decay = lambda epoch: 1 + (decay_ratio - 1) * ((epoch+self.start_epoch)/decay_epochs)\
        #    if (epoch+self.start_epoch) < decay_epochs else decay_ratio
        #self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=polynomial_decay)
        #self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, **config_dict['trainer']['lr_sheduler'])
        last_epoch = self.start_epoch if self.resume_mode else -1
        #self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5, last_epoch=last_epoch)
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=range(5,30,5), gamma=0.5, last_epoch=last_epoch)
           
        if gpu_no == 0:
            #! create folders to save trained model and results
            print('************** %s [Resume:%s | MP:%s | GPU_NUM:%d]**************' %\
                (self.name, self.resume_mode, self.mpdist, gpu_num))
            print('[%s] with %d (M) parameters was created:' % (config_dict['model'], param_count//1e6))
            util.exists_or_mkdir(self.work_dir)
            util.exists_or_mkdir(self.cache_dir, need_remove=False)
            #! save config-json file to work directory
            json.dump(config_dict, open(os.path.join(self.work_dir, 'config_script.json'), "w"), indent=4, sort_keys=False)
        
        '''dataset and loss construction'''
        self.trainLoss = loss.GLoss(config_dict['trainer'], self.cache_dir, self.mpdist, self.gpu_no)
        self.train_loader = dataset.create_dataloader(config_dict['dataset']['name'], config_dict['dataset']['train'],\
            self.batch_size, need_shuffle=True, is_mpdist=self.mpdist, world_size=gpu_num, rank=gpu_no)
        
        '''dataset and loss for validation'''
        if self.need_valid and self.gpu_no == 0:
            self.val_dir = os.path.join(self.cache_dir, 'validation')
            util.exists_or_mkdir(self.val_dir, need_remove=False)
            self.val_encode_dir = os.path.join(self.val_dir, 'encode')
            self.val_decode_dir = os.path.join(self.val_dir, 'decode')
            util.exists_or_mkdir(self.val_encode_dir, need_remove=True)
            util.exists_or_mkdir(self.val_decode_dir, need_remove=True)
            self.valLoss = loss.GLoss(config_dict['trainer'], os.path.join(self.val_dir, 'val_loss'))
            self.valid_loader = dataset.create_dataloader(config_dict['dataset']['name'], config_dict['dataset']['val'],\
                self.batch_size, need_shuffle=False, is_mpdist=False)
        

    def _train(self):
        start_time = time.time()
        for epoch in range(self.start_epoch, self.n_epochs):
            start_time_epoch = time.time()
            epoch_lr = self.lr_scheduler.get_lr()[0]
            #epoch_lr = self.lr_scheduler.state_dict()['param_groups'][0]['lr']
            epochLoss = self._train_epoch(epoch) 
            self.lr_scheduler.step()
            #self.lr_scheduler.step(epochLoss)
            if self.gpu_no != 0:
                continue
            epochMetric = self._valid_epoch(epoch) if self.need_valid else 0.0
            print("[*] --- epoch: %d/%d | loss: %4.4f | metric: %4.4f | Time-consumed: %4.2f ---" % \
                (epoch+1, self.n_epochs, epochLoss, epochMetric, (time.time() - start_time_epoch)))

            #! save losses and learning rate
            self.trainLoss.save_epoch_losses(self.resume_mode)
            self.learningrateList.append(epoch_lr)
            util.save_list(os.path.join(self.cache_dir, "lr_list"), self.learningrateList, self.resume_mode)
            if ((epoch+1) % self.config_dict['trainer']['save_epochs'] == 0 or epoch == (self.n_epochs-1)):
                print('---------- saving model ...')
                self._save_checkpoint(epoch)
                #util.visualizeLossCurves(self.cache_dir, epoch)
                #util.visualizeLossCurves(os.path.join(self.val_dir, 'val_loss'), epoch)
            if (self.need_valid and self.monitorMetric > epochMetric):
                self.monitorMetric = epochMetric
                if epoch > 0.5*self.n_epochs:
                    print('---------- saving best model ...')
                    self._save_checkpoint(epoch, save_best=True)
        #! displaying the training time
        print("Training finished! consumed %f sec" % (time.time() - start_time))

        
    def _train_epoch(self, epoch):
        #! set model to training mode
        self.model.train()
        st = time.time()
        for batch_idx, sample_batch in enumerate(self.train_loader):
            #! depatch sample batch
            input_colors, target_ABs = sample_batch['colors'], sample_batch['ABs']
            input_grays = sample_batch['grays']
            #! transfer data to target device
            input_colors = input_colors.cuda(non_blocking=True)
            input_grays = input_grays.cuda(non_blocking=True)
            target_ABs = target_ABs.cuda(non_blocking=True)
            et = time.time()
            #pdb.set_trace()
            #! reset gradient buffer to zero
            self.optimizer.zero_grad()
            #sparse_loss, prob_maps, gate_maps = self.model(input_colors)
            InvL = self.model(input_colors)
            color_seeds = target_ABs 
           
            hint_masks = torch.ones_like(InvL)
            pred_ABs = self.colorization(InvL, color_seeds, hint_masks)

            sparse_loss = 0
            pred_color = torch.cat((target_ABs, input_grays), dim=1)
            #data = {'pred_ABs':pred_ABs, 'target_ABs':target_ABs, 'sparse_loss':0, 'pred_gray':InvL}
            data = {'target_color':input_colors, 'pred_color':pred_color, 'sparse_loss':0, 'pred_gray':InvL}
            totalLoss_idx = self.trainLoss(data, epoch)
            totalLoss_idx.backward()
            self.optimizer.step()
            #! add to epoch losses
            totalLoss = totalLoss_idx.item()
            # iteration information
            if self.gpu_no == 0 and epoch == self.start_epoch and batch_idx == 0:
                print('@@@start loss:%4.4f' % totalLoss_idx.item())
            if self.gpu_no == 0 and (batch_idx+1) % self.config_dict['trainer']['display_iters'] == 0:
                tm = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
                print("%s >> [%d/%d] iter:%d loss:%4.4f [data/base:%4.3f%%]" % \
                    (tm, epoch+1, self.n_epochs, batch_idx+1, totalLoss_idx.item(), 100*(et-st)/(time.time()-et)))
            st = time.time()
        #! record epoch average loss
        epoch_loss = self.trainLoss.get_epoch_losses()
        return epoch_loss

        
    def _valid_epoch(self, epoch):
        #! set model to evaluation mode
        self.model.eval()
        with torch.no_grad():
            for batch_idx, sample_batch in enumerate(self.valid_loader):
                #! depatch sample list
                input_colors, target_ABs = sample_batch['colors'], sample_batch['ABs']
                input_grays = sample_batch['grays']
                #! transfer data to target device
                input_colors = input_colors.cuda(non_blocking=True)
                input_grays = input_grays.cuda(non_blocking=True)
                target_ABs = target_ABs.cuda(non_blocking=True)
                #! forward process
                #sparse_loss, prob_maps, gate_maps = self.model(input_colors)
                InvL = self.model(input_colors)
                color_seeds = target_ABs
                hint_masks = torch.ones_like(InvL)
                pred_ABs = self.colorization(InvL, color_seeds, hint_masks)
                sparse_loss = 0
                pred_color = torch.cat((target_ABs, input_grays), dim=1)
                #data = {'pred_ABs':pred_ABs, 'target_ABs':target_ABs, 'sparse_loss':0, 'pred_gray':InvL}
                data = {'target_color':input_colors, 'pred_color':pred_color, 'sparse_loss':0, 'pred_gray':InvL}
                #color_seeds = target_ABs
                #hint_masks = gate_maps
                #pred_ABs = self.colorization(input_grays, color_seeds, hint_masks)
                #data = {'pred_ABs':pred_ABs, 'target_ABs':target_ABs, 'sparse_loss':sparse_loss}
                self.valLoss(data, 0)
                #name_list = self.valid_filenames[cnt*self.batch_size:(cnt+1)*self.batch_size]
                #! save color images
                rgb_tensor = basic.lab2rgb(torch.cat((input_grays,pred_ABs), 1))
                rgb_imgs = basic.tensor2array(rgb_tensor*2.0 - 1.0)
                #hint_masks = basic.tensor2array(gate_maps*2.0 - 1.0)
                #dilated_hints = basic.mark_color_hints(input_grays, target_ABs, gate_maps, kernel_size=3)
                #hint_masks = basic.tensor2array(dilated_hints*2.0 - 1.0)
                #util.save_images_from_batch(hint_masks, self.val_encode_dir, None, batch_idx)
                util.save_images_from_batch(rgb_imgs, self.val_decode_dir, None, batch_idx)
            #! average metric
            epochMetric = self.valLoss.get_epoch_losses()
            self.valLoss.save_epoch_losses(self.resume_mode)
        return epochMetric
            

    def _save_checkpoint(self, epoch, save_best=False):
        state = {
            'epoch': epoch,
            'monitor_best': self.monitorMetric,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        save_path = os.path.join(self.work_dir, 'model_last.pth.tar')
        if save_best:
            save_path = os.path.join(self.work_dir, 'model_best.pth.tar')
        #! save checkpoint
        torch.save(state, save_path)


    def _resume_checkpoint(self):
        resume_path = os.path.join(self.work_dir, 'model_last.pth.tar')
        if os.path.isfile(resume_path) is False:
            print("@@@Warning:", resume_path, " is invalid checkpoint location & traning from scratch ...")
            return False
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.monitor_best = checkpoint['monitor_best']
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print('[*] checkpoint (epoch_%d) loaded successfully.'% self.start_epoch)
        return True


def main_worker(gpu_no, world_size, config_dict, resume_mode):
    gpu_num = world_size
    dist.init_process_group(                                   
        backend='nccl',
        init_method='env://',
        world_size=gpu_num,
        rank=gpu_no
    )
    torch.cuda.set_device(gpu_no)
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_no)
    node = Trainer(config_dict, resume_mode, True, gpu_num, gpu_no)
    node._train()

        
if __name__ == '__main__':
    print("FLAG: %s" % datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mp', action='store_true', help='multi-proc parallel or not')
    parser.add_argument('--resume', action='store_true', help='resume from checkpoint or not')
    parser.add_argument('--config_path', type=str, default='./PredictiveCoding_script.json', help='path of configure file')
    args = parser.parse_args()
    config_dict = json.load(open(args.config_path))
    gpu_num = torch.cuda.device_count()
    if args.mp:
        print("<< Distributed Training with ", gpu_num, " GPUS/Processes. >>")
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '8888'
        mp.spawn(main_worker, nprocs=gpu_num, args=(gpu_num, config_dict, args.resume))
    else:
        node = Trainer(config_dict, resume=args.resume, mpdist=False, gpu_num=gpu_num)
        node._train()
