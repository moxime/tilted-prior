import torch
import torch.nn as nn
from torch.autograd import Variable
import util

class Loss(nn.Module):

    def __init__(self, loss_type, tilt, nz):
        super(Loss, self).__init__()
        if tilt != None:
            print('optimizing for min kld')
            self.mu_star = util.kld_min(tilt, nz)
            print('mu_star: {:.3f}'.format(self.mu_star))
        else:
            self.mu_star = None

        self.nz = nz
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.loss_type = loss_type 
        if not loss_type in ('mse', 'rmse', 'cross_entropy'):
            raise ValueError('{} is not a valid loss type, choose either (r)mse or cross_entropy')

    def forward(self, x, x_out, mu, logvar, ood=False):
        # recon loss options
        if self.loss_type in ('mse', 'rmse'):
            
            # recon = torch.linalg.norm(x - x_out, dim=(1,2,3))
            recon = torch.sum(torch.square(x - x_out), dim=(1,2,3))
            if self.loss_type == 'rmse':
                recon = recon.sqrt()
            if not ood: # batch support for aucroc testing
                recon = torch.mean(recon) 
        elif self.loss_type == 'cross_entropy':    
            b = x.size(0)
            target = Variable(x.data  * 255).long()
            out = x_out.permute(0, -1, -4, -3, -2)
            recon = self.ce_loss(out, target).sum((-3, -2, -1))
            if not ood:
                recon = torch.sum(recon) / b
            # else: # batch support for aucroc testing
            #     print(*x.shape, *x_out.shape, *recon.shape)
            #     print('not implimented yet')
            #     import sys
            #     sys.exit()

        # kld loss options
        if self.mu_star == None:
            kld = -1/2*torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), (1))
        else:
            mu_norm = torch.linalg.norm(mu, dim=1)
            kld = 1/2*torch.square(mu_norm - self.mu_star)
        if not ood:
            kld = torch.mean(kld)

        return recon, kld
