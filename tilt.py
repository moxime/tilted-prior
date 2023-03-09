from importlib_metadata import requires
import os, sys, argparse, util, model, torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.utils as utils
import torch.optim as optim
from sklearn import metrics
from sklearn.decomposition import PCA
plt.style.use('seaborn-whitegrid')

# plots recon for ood close to sphere (prior)
def show_outlier(x, x_out, kld, r, threshold):
    for j in range(kld.shape[0]):
        if kld[j] <= threshold:
            print(r[j])
            fig = plt.figure()
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)

            ax1.imshow(x[j].detach().cpu().numpy()[0], cmap='gray')
            ax2.imshow(x_out[j].detach().cpu().numpy()[0], cmap='gray')
            plt.show()

# show latent point 
def show_latent(z_train, z_test, z_ood, j):
    c = ['k', 'dodgerblue', 'mediumseagreen']
    m = ['o', '^', 'x']
    s = [9, 14, 14]
    plt.figure(figsize=(4,4))

    for i, z in enumerate([z_train, z_test, z_ood]):
        # plot points
        n = 3000
        z = z.detach().cpu().numpy()

        # use pca subspace
        pca = PCA()
        out = pca.fit(z)
        v = out.components_[:2]

        # get projections
        x1 = z @ v[0]
        x2 = z @ v[1]
        x = np.vstack((x1, x2)).T
        
        # correct for distances
        zd = np.linalg.norm(z, axis=1)
        xd = np.linalg.norm(x, axis=1)
        x1 = (zd / xd) * x1
        x2 = (zd / xd) * x2

        x = np.vstack((x1, x2)).T

        plt.scatter(x1[:n], x2[:n], s=s[i], color=c[i], marker=m[i])
    
    lim = 35
    plt.ylim([-lim, lim])
    plt.xlim([-lim, lim])

    if j == 0: plt.legend(['Train', 'Test - ID', 'Test - OOD'])
    out = np.stack((z_train.cpu().detach().numpy(), z_test.cpu().detach().numpy(), z_ood.cpu().detach().numpy()))
    #print(out.shape)
    #np.save('latent/z_{}'.format(j), out)
    plt.tight_layout()
    # plt.savefig('latent/img {}.pdf'.format(j))
    plt.close()

# main backprop loop
def nest(all_loaders, nets, loss_fns, dist, device='cuda'):
    ood_loader, test_loader, train_loader = all_loaders
    netE, netD = nets
    in_loss_fn, out_loss_fn = loss_fns

    # backprop samples before test
    for i, (x_ood, _) in enumerate(ood_loader):   
        # get in dist test batch (not seen during training)
        x_in, _ = next(iter(test_loader))
        x_in = x_in.to(device)
        x_ood = x_ood.to(device)

        # backprop
        optimizer1 = optim.Adam(netE.parameters(), lr=5e-5, betas=(0.9, 0.999), weight_decay=3e-5) 
        optimizer2 = optim.Adam(netD.parameters(), lr=5e-5, betas=(0.9, 0.999), weight_decay=3e-5) 

        # main backprop loop 
        ne = 25 if n == 6 else 5
        for j in range(ne):
            # regular train pass
            x_train, _ = next(iter(train_loader))
            x_train = x_train.to(device)

            optimizer1.zero_grad()
            optimizer2.zero_grad()

            # train (normal)
            z_train, mu_train, logvar_train = netE(x_train)
            x_out_train = netD(z_train)
            recon_train, kld_train = in_loss_fn(x_train, x_out_train, mu_train, logvar_train)
            loss = 1 * (recon_train + kld_train)
        
            # ood pass
            opt_beta = 0.1
            z_ood, mu_ood, logvar_ood = netE(x_ood)
            x_out_ood = netD(z_ood)
            mu_ood = mu_ood - dist * torch.ones_like(mu_ood).to(device)
            recon_ood, kld_ood = out_loss_fn(x_ood, x_out_ood, mu_ood, logvar_ood)
            loss += opt_beta * (recon_ood + kld_ood)

            # in dist pass
            z_in, mu_in, logvar_in = netE(x_in)
            x_out_in = netD(z_in) # TODO: out in isn't good naming 
            recon_in, kld_in = out_loss_fn(x_in, x_out_in, mu_in, logvar_in)
            loss += opt_beta * (recon_in + kld_in)

            # optimize
            loss.backward()
            optimizer1.step()
            optimizer2.step()

            # visualize
            if j == 0: show_latent(mu_train, mu_in, mu_ood, i)
        if i >= 50: break 
    return netE, netD

if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_name', default=None, help='path to model checkpoint')
    parser.add_argument('--dataroot', default='../ood/data', help='path to dataset')
    parser.add_argument('--nest', default=1, type=int, help='0 no nest, 1 then nest')
    parser.add_argument('--ood_tilt', default=None, type=float, help='prior to use for nesting trick')
    parser.add_argument('--dist', default=0.0, type=float, help='distance from origin on each dim (for ood prior)')

    parser.add_argument('--samples', type=int, default=200, help='number of samples for OOD testing')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--aucroc', type=bool, default=True, help='boolean, run aucroc testing')
    parser.add_argument('--device')
    
    opt = parser.parse_args()

    if opt.test_name == None:
        raise ValueError('enter a load folder')
 
    cudnn.benchmark = True
    device = opt.device if torch.cuda.is_available() else 'cpu'
    print(device)
    
    # information from test_name files
    load_path = os.path.join('results', opt.test_name)
    [train_dataset, loss_type, tilt, nz]= util.get_test_info(load_path)
    tilt = torch.tensor(float(tilt)) if tilt != None else None
    if tilt is None and opt.dist == 0 and opt.nest == 1:
        print('error: using tilt and dist = 0')
        sys.exit()

    # display data, convert to useable datatypes
    print('latent dims: {}, tilt: {}'.format(nz, tilt))

    # load datasets
    loaders,loader_names,image_size = util.load_datasets(train_dataset, opt.dataroot,
                                                         batch_size=opt.batch_size, num_workers=opt.workers)    
    train_loader = loaders[0] 
    test_loader = loaders[1] 
    print(*loader_names)

    # loss function
    in_loss_fn = model.Loss(loss_type, tilt, nz) 
    out_loss_fn = model.Loss(loss_type, opt.ood_tilt, nz) 

    # make model, get state dict
    netE = model.Encoder(image_size, nz, tilt)
    netD = model.Decoder(image_size, nz, loss_type)
    state_E = torch.load(os.path.join(load_path, 'encoder.pth'), map_location=device)
    state_D = torch.load(os.path.join(load_path, 'decoder.pth'), map_location=device)

    print('going', len(loaders))
    img_dir = os.path.join(load_path, 'samples_from_ood_testing')
    for n, ood_loader in enumerate(loaders):
        # load original params
        netE.load_state_dict(state_E)
        netD.load_state_dict(state_D)
        netE.to(device)
        netD.to(device)       
        #        if n <= 3: continue
        print('testing dataset: {} of len {}'.format(loader_names[n], len(ood_loader) * opt.batch_size))

        # main backprop loop
        # return upated network 
        if opt.nest == 1:
            print('nesting')
            all_loaders = (ood_loader, test_loader, train_loader)
            nets = (netE, netD)
            loss_fns = (in_loss_fn, out_loss_fn)
            netE, netD = nest(all_loaders, nets, loss_fns, opt.dist)
        else: print('skipping nesting')

        # main test loop
        all_track = []
        eval_loaders = (ood_loader, test_loader)
        for n, loader in enumerate(eval_loaders):
            for i, (x, _) in enumerate(loader):   
                if i > 20: break
                x = x.to(device)

                z, mu, logvar = netE(x)
                x_out = netD(z)
                recon, kld = in_loss_fn(x, x_out, mu, logvar, ood=True)
                loss = (recon + kld).detach().cpu().numpy()
                # print('*** x:', *x.shape, 'x_out:', *x_out.shape, 'loss', loss)
                if i == 0: track = loss
                else: track = np.concatenate((track, loss))
            all_track.append(track)

        # NOTE: assumes same number of id and ood samples
        scores = np.concatenate((all_track[0], all_track[1]))
        labels = np.concatenate((np.ones_like(all_track[0]), np.zeros_like(all_track[1])))
        fpr, tpr, _ = metrics.roc_curve(labels, scores)
        for t, f in zip(tpr, fpr):
            if t > 0.95: break
    
        auc = metrics.auc(fpr, tpr)
        print('auc: {:.1%} fpr@{:.1%}={:.1%}'.format(auc, t, f))
        print()

        #if opt.aucroc:
        #    util.aucroc(opt.test_name, 'tilt', loader_names, train_dataset)

            #if i == 0:
                #recon_track = recon.cpu().numpy()
                #kld_track = kld.cpu().numpy()
            #else:
                #recon_track = np.concatenate((recon_track, recon.cpu().numpy()))
                #kld_track = np.concatenate((kld_track, kld.cpu().numpy()))

            #if i >= opt.samples or i == len(loader)-1: #test for _ batches
                #save_path = os.path.join(load_path, 'aucroc', 'tilt')
                #os.makedirs(save_path, exist_ok=True)
                #np.save(os.path.join(save_path, loader_names[n] + ' recon_score.npy'), recon_track) 
                #np.save(os.path.join(save_path, loader_names[n] + ' kld_score.npy'), kld_track)
                #break        



#########
# old code
#########

## for showing outliers
#r, q = loss_fn(x_a, x_out_a, mu_a, logvar_a, ood=True)
#print(torch.mean(r) + 3 * torch.std(r))
#r_temp, kld_temp = loss_fn(x, x_out, mu, logvar, ood=True)
#show_outlier(x, x_out, kld_temp, r_temp, threshold=0.016)
