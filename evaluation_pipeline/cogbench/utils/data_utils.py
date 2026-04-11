from tqdm import tqdm
import h5py
import torch
import numpy as np
import scipy.io as scio
import itertools as itools
from functools import reduce

zs = lambda v: (v-v.mean(0))/v.std(0)

def mult_diag(d, mtx, left=True):
    """
    the code is adapted from https://github.com/HuthLab/speechmodeltutorial
    Multiply a full matrix by a diagonal matrix.
    This function should always be faster than dot.
    Input:
        d -- 1D (N,) array (contains the diagonal elements)
        mtx -- 2D (N,N) array

    Output:
        mult_diag(d, mts, left=True) == dot(diag(d), mtx)
        mult_diag(d, mts, left=False) == dot(mtx, diag(d))
    
    By Pietro Berkes
    From http://mail.scipy.org/pipermail/numpy-discussion/2007-March/026807.html
    """
    if left:
        # return (d*mtx.T).T
        return (d*mtx.transpose()).transpose()
    else:
        return d*mtx

def load_fmri(fmri_path, story_amount=60, language='zh'):
    '''
    Return:
        train_fmri: (n_TRs, n_voxels); 
        starts: the start index of each story in the fmri_path;
    '''
    train_fmri = torch.tensor([])
    starts = [0]
    if language == 'zh':
        for i in tqdm(range(1, story_amount+1), desc='loading fmri from '+fmri_path+' ...'):
            fmri_file = fmri_path+'/story_'+str(i)+'.mat'
            data =scio.loadmat(fmri_file)
            single_fmri = np.array(data['fmri_response'].T)
            train_fmri = torch.cat([train_fmri, torch.FloatTensor(single_fmri)])
            starts.append(train_fmri.shape[0])
    
    elif language == 'en':
        for i in tqdm(range(1, story_amount+1), desc='loading fmri from...'+fmri_path+' ...'):
            filepath = fmri_path+'/story_'+str(i)+'.mat'
            data = scio.loadmat(filepath)
            single_fmri = np.array(data['fmri_response'].T)
            train_fmri = torch.cat([train_fmri, torch.FloatTensor(single_fmri)])
            starts.append(train_fmri.shape[0])

    else:
        raise('Unknown language!')
        
    return train_fmri, starts

def load_feature(feature_path, story_amount=60, language='zh'):
    '''
    Return:
        train_feature: (n_TRs, feature_dim)
    '''
    train_feature = torch.tensor([])
    starts = [0]
    if language == 'zh':
        for i in tqdm(range(1, story_amount+1), desc='loading stimulus from '+feature_path+' ...'):
            feature_file = feature_path + '/story_'+str(i)+'.mat'
            data = h5py.File(feature_file, 'r')
            single_feature = torch.tensor(np.array(data['word_feature'])).transpose(0, 1)
            
            train_feature = torch.cat([train_feature, single_feature])
            starts.append(train_feature.shape[0])
    elif language == 'en':
        for i in tqdm(range(1, story_amount+1), desc='loading stimulus from '+feature_path+' ...'):
            filepath = feature_path+'/story_%d.mat'%i
            data = h5py.File(filepath, 'r')
            single_feature = torch.tensor(np.array(data['word_feature'])).transpose(0, 1)
            train_feature = torch.cat([train_feature, single_feature])
            starts.append(train_feature.shape[0])
    else:
        raise('Unknown language!')
        
    return train_feature, starts

def ridge_multidim(
    train_fmri,
    train_feature,
    valid_fmri,
    valid_feature,
    alphas,
    use_cuda=False,
    singcutoff=1e-10,
):
    """
    this function can be used on features with more than 1 dimension, 
    such as word embeddings (BERT, elmo, etc.), pos tags, 
    or other semantic features
    """
    
    if use_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    train_feature = train_feature.to(device)
    train_fmri = train_fmri.to(device)
    valid_feature = valid_feature.to(device)
    valid_fmri = valid_fmri.to(device)

    U, S, Vh = torch.linalg.svd(train_feature, full_matrices=False)
    V = Vh.transpose(0, 1)
    
    ngoodS = torch.sum(S>singcutoff)
    U = U[:,:ngoodS]
    S = S[:ngoodS]
    V = V[:,:ngoodS]

    alphas = torch.tensor(alphas, device=device, dtype=train_feature.dtype)
    UR = torch.matmul(U.transpose(0, 1), train_fmri)
    PVh = torch.matmul(valid_feature, V)

    zvalid_fmri = zs(valid_fmri)
    Rcorrs = [] ## Holds training correlations for each alpha
    for a in alphas:
        D = S/(S**2+a**2) ## Reweight singular vectors by the ridge parameter
        pred = torch.matmul(mult_diag(D, PVh, left=False), UR)
        Rcorr = (zvalid_fmri*zs(pred)).mean(0)                
        Rcorr[torch.isnan(Rcorr)] = 0
        Rcorrs.append(Rcorr)
    
    return Rcorrs

def ridge_nested_cv(total_fmri, total_feature, result_dir, sub, use_cuda=None):
    """ 
    nested cross-validation, which is applicable to situations without
    designated test set
    Return: corrs on all out test set
    """
    if use_cuda is None:
        use_cuda = torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    print(f"ridge_nested_cv device: {device}")

    alphas = np.logspace(-3, 3, 10)
    nTR, n_dim = total_feature.shape
    foldlen = int(nTR/5)
    test_corrs = []
    weights = []
    inds = list(range(nTR))
    # import pdb;pdb.set_trace()
    for fold in tqdm(range(5), desc='doing nested ridge regression...'):
        test_inds = inds[fold*foldlen:(fold+1)*foldlen]
        inner_inds = inds[0:fold*foldlen]+inds[(fold+1)*foldlen:]
        infoldlen = int(len(inner_inds)/5)
        test_fmri = total_fmri[test_inds]
        test_feature = total_feature[test_inds]

        val_corrs = []
        for infold in range(5):
            val_inds = inner_inds[infold*infoldlen:(infold+1)*infoldlen]
            train_inds = inner_inds[0:infold*infoldlen]+inner_inds[(infold+1)*infoldlen:]
            train_fmri = total_fmri[train_inds]
            valid_fmri = total_fmri[val_inds]
            train_feature = total_feature[train_inds]
            valid_feature = total_feature[val_inds]
            Rcorrs = ridge_multidim(
                train_fmri,
                train_feature,
                valid_fmri,
                valid_feature,
                alphas,
                use_cuda=use_cuda,
            )
            '''if n_dim == 1:
                Rcorrs = self.ridge_1dim(train_fmri, train_feature, valid_fmri, valid_feature, self.alphas)
            else:
                Rcorrs = self.ridge_multidim(train_fmri, train_feature, valid_fmri, valid_feature, \
                    self.alphas, self.cuda0, self.cuda1, self.use_cuda)'''
            val_corrs.append(torch.stack(Rcorrs))
        val_corrs = torch.stack(val_corrs)  
        max_ind = torch.argmax(val_corrs.mean(2).mean(0))
        bestalpha = alphas[max_ind]  
        inner_feature = total_feature[inner_inds].to(device)
        inner_fmri = total_fmri[inner_inds].to(device)
        test_feature = test_feature.to(device)
        test_fmri = test_fmri.to(device)

        U, S, Vh = torch.linalg.svd(inner_feature, full_matrices=False)
        V = Vh.transpose(0, 1)
        UR = torch.matmul(U.transpose(0, 1), inner_fmri)
        bestalpha_t = torch.tensor(bestalpha, dtype=S.dtype, device=device)
        wt = reduce(torch.matmul, [V, torch.diag(S/(S**2+bestalpha_t**2)), UR])
        pred = torch.matmul(test_feature, wt)
        corrs = (zs(pred)*zs(test_fmri)).mean(0)
        test_corrs.append(corrs)
        weights.append(wt)
    test_corrs = torch.stack(test_corrs)
    weights = torch.stack(weights)
    savefile = result_dir+sub+'_average.mat'
    scio.savemat(savefile, {'test_corrs':np.array(test_corrs.mean(0).cpu())})
    return np.array(test_corrs.mean(0).cpu())


def ridge_train_dev_test(
    train_fmri,
    train_feature,
    dev_fmri,
    dev_feature,
    test_fmri,
    test_feature,
    result_dir,
    sub,
    use_cuda=None,
):
    """Ridge regression with explicit train/dev/test split."""
    if use_cuda is None:
        use_cuda = torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    print(f"ridge_train_dev_test device: {device}")

    alphas = np.logspace(-3, 3, 10)

    dev_corrs_by_alpha = ridge_multidim(
        train_fmri,
        train_feature,
        dev_fmri,
        dev_feature,
        alphas,
        use_cuda=use_cuda,
    )
    dev_corrs_by_alpha = torch.stack(dev_corrs_by_alpha)
    max_ind = torch.argmax(dev_corrs_by_alpha.mean(1))
    bestalpha = alphas[max_ind]

    # Keep dev strictly as validation/evaluation split.
    fit_feature = train_feature.to(device)
    fit_fmri = train_fmri.to(device)
    test_feature = test_feature.to(device)
    test_fmri = test_fmri.to(device)

    U, S, Vh = torch.linalg.svd(fit_feature, full_matrices=False)
    V = Vh.transpose(0, 1)
    UR = torch.matmul(U.transpose(0, 1), fit_fmri)
    bestalpha_t = torch.tensor(bestalpha, dtype=S.dtype, device=device)
    wt = reduce(torch.matmul, [V, torch.diag(S/(S**2+bestalpha_t**2)), UR])

    pred = torch.matmul(test_feature, wt)
    corrs = (zs(pred) * zs(test_fmri)).mean(0)
    corrs[torch.isnan(corrs)] = 0

    savefile = result_dir + sub + '_average.mat'
    scio.savemat(
        savefile,
        {
            'test_corrs': np.array(corrs.detach().cpu()),
            'best_alpha': np.array([bestalpha], dtype=np.float32),
        },
    )
    return np.array(corrs.detach().cpu())