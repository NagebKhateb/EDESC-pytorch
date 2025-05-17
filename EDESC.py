from __future__ import print_function, division
import argparse
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader
from utils import LoadDataset, cluster_acc
import warnings
from AutoEncoder import AE
from InitializeD import Initialization_D
from Constraint import D_constraint1, D_constraint2
import time
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

   
class EDESC(nn.Module):

    def __init__(self,
                 n_enc_1,
                 n_enc_2,
                 n_enc_3,
                 n_dec_1,
                 n_dec_2,
                 n_dec_3,
                 n_input,
                 n_z,
                 n_clusters,
                 num_sample,
                 pretrain_path='data/reuters.pkl'):
        super(EDESC, self).__init__()
        self.pretrain_path = pretrain_path
        self.n_clusters = n_clusters

        self.ae = AE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,
            n_input=n_input,
            n_z=n_z)	

        # Subspace bases proxy
        self.D = Parameter(torch.Tensor(n_z, n_clusters))

        
    def pretrain(self, path=''):
        if path == '':
            pretrain_ae(self.ae)
        # Load pre-trained weights
        self.ae.load_state_dict(torch.load(self.pretrain_path, map_location='cpu'))
        print('Load pre-trained model from', path)

    def forward(self, x):
        
        x_bar, z = self.ae(x)
        d = args.d
        s = None
        eta = args.eta
      
        # Calculate subspace affinity
        for i in range(self.n_clusters):	
			
            si = torch.sum(torch.pow(torch.mm(z,self.D[:,i*d:(i+1)*d]),2),1,keepdim=True)
            if s is None:
                s = si
            else:
                s = torch.cat((s,si),1)   
        s = (s+eta*d) / ((eta+1)*d)
        s = (s.t() / torch.sum(s, 1)).t()
        return x_bar, s, z

    def total_loss(self, x, x_bar, z, pred, target, dim, n_clusters, beta):

	# Reconstruction loss
        reconstr_loss = F.mse_loss(x_bar, x)     
        
        # Subspace clustering loss
        kl_loss = F.kl_div(pred.log(), target)
        
        # Constraints
        d_cons1 = D_constraint1()
        d_cons2 = D_constraint2()
        loss_d1 = d_cons1(self.D)
        loss_d2 = d_cons2(self.D, dim, n_clusters)
  
        # Total_loss
        total_loss = reconstr_loss + beta * kl_loss + loss_d1 + loss_d2

        return total_loss

		
def refined_subspace_affinity(s):
    weight = s**2 / s.sum(0)
    return (weight.t() / weight.sum(1)).t()

def pretrain_ae(model):

    train_loader = DataLoader(dataset, batch_size=512, shuffle=True)
    print(model)
    optimizer = Adam(model.parameters(), lr=args.lr)
    for epoch in range(50):
        total_loss = 0.
        for batch_idx, (x, _, _) in enumerate(train_loader):
            x = x.to(device)
            optimizer.zero_grad()
            x_bar, z = model(x)
            loss = F.mse_loss(x_bar, x)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        print("Epoch {} loss={:.4f}".format(epoch,
                                            total_loss / (batch_idx + 1)))
        torch.save(model.state_dict(), args.pretrain_path)
    print("Model saved to {}.".format(args.pretrain_path))

plt.ion()
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
fig.patch.set_facecolor('#f5f5f5')

def live_plot(acc_list, nmi_list, max_acc_list, max_nmi_list):
    ax.clear()
    ax.set_facecolor('#f9f9f9')
    
    ax.plot(acc_list, label='Current Accuracy', color='#1f77b4', linewidth=2)
    ax.plot(max_acc_list, label='Max Accuracy', linestyle='--', color='#1f77b4', linewidth=2)
    ax.plot(nmi_list, label='Current NMI', color='#2ca02c', linewidth=2)
    ax.plot(max_nmi_list, label='Max NMI', linestyle='--', color='#2ca02c', linewidth=2)
        
    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold', fontname='Segoe UI')
    ax.set_ylabel('Score', fontsize=14, fontweight='bold', fontname='Segoe UI')
    ax.set_title('Clustering Performance per Epoch', fontsize=16, fontweight='bold', fontname='Segoe UI')
    ax.legend(frameon=False, fontsize=12, loc='lower right')
    ax.grid(True, linestyle=':', linewidth=0.7, alpha=0.7)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout()
    plt.pause(2)

def train_EDESC():
    acc_list = []
    nmi_list = []
    max_acc_list = []
    max_nmi_list = []

    accmax = 0
    nmimax = 0

    model = EDESC(
        n_enc_1=500,
        n_enc_2=500,
        n_enc_3=1000,
        n_dec_1=1000,
        n_dec_2=500,
        n_dec_3=500,
        n_input=args.n_input,
        n_z=args.n_z,
        n_clusters=args.n_clusters,
        num_sample = args.num_sample,
        pretrain_path=args.pretrain_path).to(device)
    start = time.time()      

    # Load pre-trained model
    model.pretrain('reuters.pkl')
    optimizer = Adam(model.parameters(), lr=args.lr)
    
    # Cluster parameter initiate
    data = dataset.x
    y = dataset.y
    data = torch.Tensor(data).to(device)
    x_bar, hidden = model.ae(data)
    kmeans = KMeans(n_clusters=args.n_clusters, n_init=10) 

    # Get clusters from Consine K-means 
    # ~ X = hidden.data.cpu().numpy()
    # ~ length = np.sqrt((X**2).sum(axis=1))[:,None]
    # ~ X = X / length
    # ~ y_pred = kmeans.fit_predict(X)
 
    # Get clusters from K-means
    y_pred = kmeans.fit_predict(hidden.data.cpu().numpy())
    print("Initial Cluster Centers: ", y_pred)
    
    # Initialize D
    D = Initialization_D(hidden, y_pred, args.n_clusters, args.d)
    D = torch.tensor(D).to(torch.float32)
    accmax = 0
    nmimax = 0  
    y_pred_last = y_pred
    model.D.data = D.to(device)
    
    model.train()
    
    for epoch in range(200):
        x_bar, s, z = model(data)

        # Update refined subspace affinity
        tmp_s = s.data
        s_tilde = refined_subspace_affinity(tmp_s)

        # Evaluate clustering performance
        y_pred = tmp_s.cpu().detach().numpy().argmax(1)
        delta_label = np.sum(y_pred != y_pred_last).astype(
            np.float32) / y_pred.shape[0]
        y_pred_last = y_pred
        acc = cluster_acc(y, y_pred)
        nmi = nmi_score(y, y_pred)

        acc_list.append(acc)
        nmi_list.append(nmi)
        max_acc_list.append(accmax)
        max_nmi_list.append(nmimax)

        if acc > accmax:
            accmax = acc
        if nmi > nmimax:
            nmimax = nmi            
        print('Iter {}'.format(epoch), ':Current Acc {:.4f}'.format(acc),
                  ':Max Acc {:.4f}'.format(accmax),', Current nmi {:.4f}'.format(nmi), ':Max nmi {:.4f}'.format(nmimax))
        
        live_plot(acc_list, nmi_list, max_acc_list, max_nmi_list)

        ############## Total loss function ######################
        loss = model.total_loss(data, x_bar, z, pred=s, target=s_tilde, dim=args.d, n_clusters = args.n_clusters, beta = args.beta)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    end = time.time()
    print('Running time: ', end-start)
    return accmax, nmimax
 

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='EDESC training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--n_clusters', default=4, type=int)
    parser.add_argument('--d', default=5, type=int)
    parser.add_argument('--n_z', default=20, type=int)
    parser.add_argument('--eta', default=5, type=int)
    #parser.add_argument('--batch_size', default=512, type=int)    
    parser.add_argument('--dataset', type=str, default='reuters')
    parser.add_argument('--pretrain_path', type=str, default='data/reuters')
    parser.add_argument('--beta', default=0.1, type=float, help='coefficient of subspace affinity loss')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")
    args.dataset = 'reuters'
    if args.dataset == 'reuters':
        args.pretrain_path = 'data/reuters.pkl'
        args.n_clusters = 4
        args.n_input = 2000
        args.num_sample = 10000
        dataset = LoadDataset(args.dataset)   
    print(args)
    bestacc = 0 
    bestnmi = 0
    for i in range(10):
        acc, nmi = train_EDESC()
        if acc > bestacc:
            bestacc = acc
        if nmi > bestnmi:
            bestnmi = nmi
    print('Best ACC {:.4f}'.format(bestacc), ' Best NMI {:4f}'.format(bestnmi))
