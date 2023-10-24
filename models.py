import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class PGCH(nn.Module):

    def __init__(self, bits = 128,text_len= 1386 ):
        super(PGCH, self).__init__()


        self.bits = bits
        self.text_len = text_len


        self.act = nn.Tanh()

        self.alexnet = torchvision.models.alexnet(pretrained=True)
        self.alexnet.classifier = nn.Sequential(*list(self.alexnet.classifier.children())[:6])

        self.txt_net = nn.Sequential(
            nn.Linear(self.text_len, 512),
            nn.BatchNorm1d(512),
            nn.Tanh(),
        )

        self.img_net = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.BatchNorm1d(2048),
            nn.Tanh(),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.Tanh(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.Tanh()
        )

        self.img_text = nn.Sequential(
            nn.Linear(512, self.bits),
            nn.BatchNorm1d(self.bits),
            nn.Tanh(),
        )



        self.Discriminator_img=nn.Sequential(
                    nn.Linear(4096, 2048),
                    nn.ReLU(),
                    nn.Linear(2048, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, 512),
                    nn.ReLU(),
                    nn.Linear(512, 1),
                    nn.Sigmoid(),)

        self.Discriminator_txt=nn.Sequential(
                    nn.Linear(text_len, int(text_len/2)),
                    nn.ReLU(),
                    nn.Linear(int(text_len/2), 1),
                    nn.Sigmoid(),
                )

        self.i2t = nn.Sequential(
            nn.Linear(4096, text_len),
            nn.ReLU(),)
        self.t2i = nn.Sequential(
            nn.Linear(text_len, 4096),
            nn.ReLU(),)



    def forward(self, img, txt):

        img= self.alexnet.features(img)
        img = img.view(img.size(0), -1)
        img = self.alexnet.classifier(img)

        img = torch.tensor(img, dtype=torch.float)

        img_common = self.img_net(img)
        txt_common = self.txt_net(txt)


        hash1 = self.img_text(img_common)
        hash2 = self.img_text(txt_common)

        txt_fake = self.i2t(img)
        img_fake = self.t2i(txt)

        txt_fake = self.Discriminator_txt(txt_fake)
        img_fake = self.Discriminator_img(img_fake)

        txt_real = self.Discriminator_txt(txt)
        img_real = self.Discriminator_img(img)


        return img_common,txt_common,hash1,hash2,img_real,img_fake,txt_real,txt_fake


class GCNLI(nn.Module):
    def __init__(self, bits):
        super(GCNLI, self).__init__()
        self.bits = bits
        self.gc_img = GraphConvolution(512, self.bits, 0.5, act=lambda x: x)

    def forward(self, img_common,adj):
        img_gcn = self.gc_img(img_common, adj)

        return img_gcn

class GCNLT(nn.Module):
    def __init__(self, bits):
        super(GCNLT, self).__init__()
        self.bits = bits
        self.gc_txt = GraphConvolution(512, self.bits, 0.5, act=lambda x: x)

    def forward(self, txt_common,adj):
        txt_gcn = self.gc_txt(txt_common, adj)

        return txt_gcn



class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.weight)

    def forward(self, input, adj):
        input = F.dropout(input, self.dropout, self.training)
        output = torch.mm(input, self.weight)
        output = torch.mm(adj, output)
        output = self.act(output)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
