import os
import numpy as np
import argparse
import torch
import util
import model
import torch.backends.cudnn as cudnn
import sys

if __name__== "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', default='../ood/data', help='path to dataset')

    parser.add_argument('--samples', type=int, default=200, help='number of samples for OOD testing')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')

    parser.add_argument('--test_name', default=None, help='path to model checkpoint')
    parser.add_argument('--aucroc', type=bool, default=True, help='boolean, run aucroc testing')
    
    opt = parser.parse_args()

    if opt.test_name == None:
        raise ValueError('enter a load folder')
 
    cudnn.benchmark = True
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(device)
    
    # information from test_name files
    load_path = os.path.join('results', opt.test_name)
    [train_dataset, loss_type, tilt, nz]= util.get_test_info(load_path)
    tilt = torch.tensor(float(tilt)) if tilt != None else None

    # display data, convert to useable datatypes
    print('latent dims: {}, tilt: {}'.format(nz, tilt))

    # load datasets
    print('loading data')
    
    loaders,loader_names,image_size = util.load_datasets(train_dataset, opt.dataroot,
                                                batch_size=opt.batch_size, num_workers=4)    
    # load network and loss
    print('loading network') 
    netE = model.Encoder(image_size, nz, tilt)
    state_E = torch.load(os.path.join(load_path, 'encoder.pth'))
    netE.load_state_dict(state_E)
    netE.to(device)

    netD = model.Decoder(image_size, nz, loss_type)
    state_D = torch.load(os.path.join(load_path, 'decoder.pth'))
    netD.load_state_dict(state_D)
    netD.to(device)
        
    loss_fn = model.Loss(loss_type, tilt, nz) 

    # main test loop
    for n, loader in enumerate(loaders): 
        print('testing dataset: {}'.format(loader_names[n]))
        
        for i, (x, _) in enumerate(loader):   
            # get negative log-likelihood from model
            with torch.no_grad():
                x = x.to(device)

                # network pass
                z, mu, logvar = netE(x)
                x_out = netD(z)

                recon, kld = loss_fn(x, x_out, mu, logvar, ood=True)
                nll = recon + kld

                if i == 0:
                    score_track = nll.cpu().numpy()
                else:
                    score_track = np.concatenate((score_track, nll.cpu().numpy()))

            if i >= opt.samples or i == len(loader)-1: #test for _ batches
                save_path = os.path.join(load_path, 'aucroc', 'tilt')
                os.makedirs(save_path, exist_ok=True)
                np.save(os.path.join(save_path, loader_names[n] + ' score.npy'), score_track) 
                break        

    if opt.aucroc:
        util.aucroc(opt.test_name, 'tilt', loader_names, train_dataset)
