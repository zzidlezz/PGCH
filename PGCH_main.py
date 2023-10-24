import os
import time
import torch
from torch.autograd import Variable
import dset
import metric
from metric import calculate_top_map, get_hash_targets
from models import PGCH, GCNLI, GCNLT
from metric import  affinity_tag_multi
import settings
import torch.nn.functional as F
torch.cuda.set_device(0)

class Session:
    def __init__(self):
        self.logger = settings.logger

        torch.manual_seed(1)
        torch.cuda.manual_seed_all(1)

        self.train_dataset = dset.MY_DATASET(train=True, transform=dset.train_transform)
        self.test_dataset = dset.MY_DATASET(train=False, database=False, transform=dset.test_transform)
        self.database_dataset = dset.MY_DATASET(train=False, database=True, transform=dset.test_transform)

        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                   batch_size=settings.BATCH_SIZE,
                                                   shuffle=True,
                                                   num_workers=settings.NUM_WORKERS,
                                                   drop_last=True)

        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset,
                                                  batch_size=settings.BATCH_SIZE,
                                                  shuffle=False,
                                                  num_workers=settings.NUM_WORKERS)

        self.database_loader = torch.utils.data.DataLoader(dataset=self.database_dataset,
                                                  batch_size=settings.BATCH_SIZE,
                                                  shuffle=False,
                                                  num_workers=settings.NUM_WORKERS)



        self.gcn = PGCH(bits=settings.CODE_LEN, text_len=1386)
        self.opt_G = torch.optim.Adam(self.gcn.parameters(), lr=0.0001)

        self.GCNI = GCNLI(bits=settings.CODE_LEN)
        self.opt_I = torch.optim.Adam(self.GCNI.parameters(), lr=0.0001)

        self.GCNT = GCNLT(bits=settings.CODE_LEN)
        self.opt_T = torch.optim.Adam(self.GCNT.parameters(), lr=0.0001)



    def train(self, epoch):
        self.gcn.cuda().train()
        self.GCNI.cuda().train()
        self.GCNT.cuda().train()

        loss_fn = torch.nn.MSELoss()
        loss_adv = torch.nn.BCELoss()

        for idx, (img, F_T, labels,_) in enumerate(self.train_loader):
            img = Variable(img).cuda()
            F_T = Variable(torch.FloatTensor(F_T.numpy()).cuda())
            labels = Variable(labels).cuda()
            hash_targets = get_hash_targets(settings.label_class, settings.CODE_LEN).cuda()
            multi_label_random_center = torch.randint(2, (settings.CODE_LEN,)).float().cuda()
            criterion = torch.nn.BCEWithLogitsLoss().cuda()
            center_sum = labels.float() @ hash_targets
            random_center = multi_label_random_center.repeat(center_sum.shape[0], 1)
            center_sum[center_sum == 0] = random_center[center_sum == 0]
            hash_center = 2 * (center_sum > 0).float() - 1
            _, aff1 = affinity_tag_multi(hash_center.cpu().numpy(), hash_center.cpu().numpy())
            aff1 = Variable(torch.Tensor(aff1)).cuda()


            self.opt_G.zero_grad()
            self.opt_I.zero_grad()
            self.opt_T.zero_grad()


            img_common,txt_common,hash1, hash2, img_real,img_fake,txt_real,txt_fake= self.gcn(img, F_T)
            hash1_gcn = self.GCNI(img_common, aff1)
            hash2_gcn = self.GCNT(txt_common, aff1)

            hash1_norm = F.normalize(hash1)
            hash2_norm = F.normalize(hash2)
            hash1_gcn_norm = F.normalize(hash1_gcn)
            hash2_gcn_norm = F.normalize(hash2_gcn)
            Lgra =(loss_fn(hash1_norm.mm(hash2_norm.t()),  hash1_gcn_norm.mm(hash2_gcn_norm.t())))

            loss4 = criterion(0.5*(hash1+1), 0.5*(hash_center+1))
            loss5 = criterion(0.5*(hash2+1), 0.5*(hash_center+1))
            Lconstr = loss4+loss5

            Binary1 = torch.sign(hash1)
            Binary2 = torch.sign(hash2)
            loss7 = loss_fn(hash1, Binary1)
            loss8 = loss_fn(hash2, Binary2)
            Lq =0.001*(loss7+loss8)

            loss_d_img_r = loss_adv(img_real, Variable(torch.ones(img_real.shape[0], 1)).cuda())
            loss_d_img_f = loss_adv(img_fake, Variable(torch.zeros(img_fake.shape[0], 1)).cuda())
            loss9 =  (loss_d_img_r + loss_d_img_f)
            loss_d_txt_r = loss_adv(txt_real, Variable(torch.ones(txt_real.shape[0], 1)).cuda())
            loss_d_txt_f = loss_adv(txt_fake, Variable(torch.zeros(txt_fake.shape[0], 1)).cuda())
            loss10 = (loss_d_txt_r + loss_d_txt_f)
            Ladv =(loss9+loss10)

            Li=0.1*torch.mean((torch.square(hash1 - hash2)))

            loss =Lconstr+Lgra+Lq+Li+Ladv

            loss.backward()
            self.opt_G.step()
            self.opt_I.step()
            self.opt_T.step()

            if (idx + 1) % (len(self.train_dataset) // settings.BATCH_SIZE / settings.EPOCH_INTERVAL) == 0:
                self.logger.info('Epoch [%d/%d], Iter [%d/%d] Total Loss: %.4f'
                    % (epoch + 1, settings.NUM_EPOCH, idx + 1, len(self.train_dataset) // settings.BATCH_SIZE,
                        loss.item()))


    def eval(self):
        self.logger.info('--------------------Evaluation: Calculate top MAP-------------------')
        self.gcn.cuda().eval()
        self.GCNI.cuda().eval()
        self.GCNT.cuda().eval()

        re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L = metric.compress(self.database_loader, self.test_loader, self.gcn, self.database_dataset, self.test_dataset)

        MAP_I2T_1000 = calculate_top_map(qu_B=qu_BI, re_B=re_BT, qu_L=qu_L, re_L=re_L, topk=1000)
        MAP_T2I_1000 = calculate_top_map(qu_B=qu_BT, re_B=re_BI, qu_L=qu_L, re_L=re_L, topk=1000)

        print('map(i->t): %3.4f, map(t->i): %3.4f' % (MAP_I2T_1000,MAP_T2I_1000))





def main():
    
    sess = Session()

    if settings.EVAL == True:
        sess.load_checkpoints()
        sess.eval()

    else:
        start_time = time.time()
        for epoch in range(settings.NUM_EPOCH):
            # train the Model
            sess.train(epoch)
            # eval the Model
            if (epoch + 1) % settings.EVAL_INTERVAL == 0:
                sess.eval()
            # save the model
            if epoch + 1 == settings.NUM_EPOCH:
                sess.save_checkpoints(step=epoch+1)
        train_time = time.time() - start_time
        print('Training time: ', train_time)
        sess.eval()


if __name__ == '__main__':
    if not os.path.exists('result'):
        os.makedirs('result')
    if not os.path.exists('log'):
        os.makedirs('log')
        
    main()
