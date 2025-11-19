import os
os.environ["KERAS_BACKEND"] = "torch"
import torch
from hgq.layers import QDense, QBatchNormalization, QUnaryFunctionLUT
from torch_scatter import scatter_add

from torch import nn

# from torch_geometric.nn.conv import GraphConv, EdgeConv, GCNConv
from .EdgeConv_HGQ import EdgeConv # need the dot so that it looks in same folder

class GraphMETNetwork(nn.Module):
    def __init__ (self, continuous_dim, cat_dim, norm, output_dim=1, hidden_dim=32, conv_depth=1):
    #def __init__ (self, continuous_dim, cat_dim, output_dim=1, hidden_dim=32, conv_depth=1):
        super(GraphMETNetwork, self).__init__()
       
        self.datanorm = norm

        self.embed_charge = nn.Embedding(3, hidden_dim//4)
        self.embed_pdgid = nn.Embedding(7, hidden_dim//4)
        

        self.embed_continuous_dense = QDense(hidden_dim//2) # output
        # self.embed_continuous_dense.build((None, continuous_dim)) # input
        # kernal / input / bias config: kq_conf, iq_conf, bq_conf
        self.embed_continuous_elu = QUnaryFunctionLUT(activation='elu') # iq_conf, oq_conf

        self.embed_categorical_dense = QDense(hidden_dim//2)
        self.embed_categorical_elu = QUnaryFunctionLUT(activation='elu')

        self.encode_all_dense = QDense(hidden_dim)
        self.encode_all_elu = QUnaryFunctionLUT(activation='elu')
        self.bn_all = QBatchNormalization(axis=-1) # kq_conf, iq_conf, bq_conf
 
        self.conv_continuous = nn.ModuleList()        
        for _ in range(conv_depth):
            # mesg = QDense(hidden_dim)
            # removed .jittable() as jocelyn impl doesn't have it, altered input as expectes in_channels/out_channels instead of nn
            conv_layer = EdgeConv(in_channels=hidden_dim, out_channels=hidden_dim) # to account for changed syntax
            # bn_layer = QBatchNormalization(axis=-1)
            # self.conv_continuous.append(nn.ModuleList([conv_layer, bn_layer]))
            self.conv_continuous.append(conv_layer) # because bn layer now inside of the edgeconv impl

        self.output_dense1 = QDense(hidden_dim//2)
        self.output_elu = QUnaryFunctionLUT(activation='elu')
        self.output_dense2 = QDense(output_dim)

        self.pdgs = [1, 2, 11, 13, 22, 130, 211]

    def forward(self, x_cont, x_cat, edge_index, batch, training = False): # by default training is false.
        # Normalize the input values within [0,1] range: pt, px, py, eta, phi, puppiWeight, pdgId, charge
        #norm = torch.tensor([1./2950., 1./2950, 1./2950, 1., 1., 1.]).to(device) 

        x_cont *= self.datanorm

        emb_cont = self.embed_continuous_dense(x_cont, training=training)
        emb_cont = self.embed_continuous_elu(emb_cont, training=training)

        emb_chrg = self.embed_charge(x_cat[:, 1] + 1)

        pdg_remap = torch.abs(x_cat[:, 0])
        for i, pdgval in enumerate(self.pdgs):
            pdg_remap = torch.where(pdg_remap == pdgval, torch.full_like(pdg_remap, i), pdg_remap)
        emb_pdg = self.embed_pdgid(pdg_remap)

        emb_cat = torch.cat([emb_chrg, emb_pdg], dim=1)
        emb_cat = self.embed_categorical_dense(emb_cat, training=training)
        emb_cat = self.embed_categorical_elu(emb_cat, training=training)

        emb = torch.cat([emb_cat, emb_cont], dim=1)
        emb = self.encode_all_dense(emb, training=training)
        emb = self.encode_all_elu(emb, training=training)
        emb = self.bn_all(emb, training=training)

        # graph convolution for continuous variables
        # for co_conv in self.conv_continuous:
            # dynamic, evolving knn
            # emb = emb + co_conv[1](co_conv[0](emb, knn_graph(emb, k=20, batch=batch, loop=True)))
            # static
            # emb = emb + co_conv[1](co_conv[0](emb, edge_index))
        for conv_layer in self.conv_continuous:
            emb = emb + conv_layer(emb, edge_index, batch, training=training)
            # bnlayer already from hgq (keras), conv_layer is jocelyn impl, need to alter
                
        # out = self.output(emb)
        out = self.output_dense1(emb, training = training)
        out = self.output_elu(out, training = training)
        out = self.output_dense2(out, training = training)
        
        return out.squeeze(-1)
    @property
    def losses(self):
        # Collect losses from all sub-modules (like QDense, etc.)
        collected_losses = []
        for module in self.modules():
            # Skip the container itself to avoid infinite recursion
            if module is self:
                continue
            
            # If a sub-layer has a 'losses' attribute (standard in Keras/HGQ layers), collect it
            if hasattr(module, 'losses'):
                collected_losses.extend(module.losses)
        
        return collected_losses
    @property
    def weights(self):
        # Collect weights from all sub-modules (like QDense, etc.)
        collected_weights = []
        for module in self.modules():
            if module is self:
                continue
            
            # If a sub-layer has a 'weights' attribute, collect it
            if hasattr(module, 'weights'):
                collected_weights.extend(module.weights)
        
        return collected_weights
        
    
# COPIED LOSS FUNCTIONS OVER FROM NET.PY

# tensor operations
def getdot(vx, vy):
    return torch.einsum('bi,bi->b',vx,vy)

def getscale(vx):
    return torch.sqrt(getdot(vx,vx))

def scalermul(a,v):
    return torch.einsum('b,bi->bi',a,v)

# loss function without response tune option
def loss_fn(weights, particles_vis, genMET, batch, scale_momentum = 128.):
    # particles_vis: (pT, px, py, eta, phi, puppiWeight, pdgId, charge)
    # momentum of the visible particles
    px = particles_vis[:,1]
    py = particles_vis[:,2]
 
    # gen MET = (px, py) of genMET
    # uT = (-1)*genMET
    true_px = (-1)*genMET[:,0] / scale_momentum
    true_py = (-1)*genMET[:,1] / scale_momentum

    # regress uT: MET = (-1)*uT
    # ML weights are [0,1]
    uTx = scatter_add(weights*px, batch)
    uTy = scatter_add(weights*py, batch)

    loss=0.5*( ( uTx - true_px)**2 + ( uTy - true_py)**2 ).mean()

    return loss


# loss function with response tune
def loss_fn_response_tune(weights, particles_vis, genMET, batch, c = 500, scale_momentum = 128.):
    # particles_vis: (pT, px, py, eta, phi, puppiWeight, pdgId, charge)
    # momentum of the visible particles
    px = particles_vis[:,1]
    py = particles_vis[:,2]
 
    # gen MET = (px, py) of genMET
    # uT = (-1)*genMET
    true_px = (-1)*genMET[:,0] / scale_momentum
    true_py = (-1)*genMET[:,1] / scale_momentum

    # regress uT: MET = (-1)*uT
    # ML weights are [0,1]
    uTx = scatter_add(weights*px, batch)
    uTy = scatter_add(weights*py, batch)

    loss=0.5*( ( uTx - true_px)**2 + ( uTy - true_py)**2 ).mean() 

    #print('loss (no corr):', loss)
    # response correction
    v_true = torch.stack((true_px,true_py),dim=1)
    v_regressed = torch.stack((uTx, uTy),dim=1)
        
    # response = getdot( v_true, v_regressed ) / getdot( v_true, v_true ) # dot product
    response = getscale(v_regressed) / getscale(v_true) # ratio of the MET scale
    
    #print('response:', response)
    #print('v_true:', getscale(v_true))
    #print('v_regressed:', getscale(v_regressed))

    #pT_thres = 0.         # calculate response only taking into account for events with genMET above threshold
    pT_thres = 50./scale_momentum
    resp_pos = torch.logical_and(response > 1., getscale(v_true) > pT_thres)
    resp_neg = torch.logical_and(response < 1., getscale(v_true) > pT_thres)
    
    c = c / scale_momentum
    
    response_term = c * (torch.sum(1 - response[resp_neg]) + torch.sum(response[resp_pos] - 1))

    #print('1 - response[resp_neg]:', 1 - response[resp_neg])
    #print('1 - response[resp_pos]:', response[resp_pos]-1)
    #print('response_term:', response_term)
    
    loss += response_term

    return loss


# loss function; loss normalized with genMETx and genMETy
def loss_fn_relative(weights, particles_vis, genMET, batch, c = 5000):
    # particles_vis: (pT, px, py, eta, phi, puppiWeight, pdgId, charge)
    # momentum of the visible particles
    px = particles_vis[:,1]
    py = particles_vis[:,2]
 
    # gen MET = (px, py) of genMET
    # uT = (-1)*genMET
    true_px = (-1)*genMET[:,0]
    true_py = (-1)*genMET[:,1]

    # regress uT: MET = (-1)*uT
    # ML weights are [0,1]
    uTx = scatter_add(weights*px, batch)
    uTy = scatter_add(weights*py, batch)

    loss=0.5*( ( (uTx - true_px) / true_px)**2 + ( (uTy - true_py) / true_py)**2 ).mean() 

    return loss


# loss function; loss normalized with genMET scale
def loss_fn_relative_genMET(weights, particles_vis, genMET, batch, c = 5000):
    # particles_vis: (pT, px, py, eta, phi, puppiWeight, pdgId, charge)
    # momentum of the visible particles
    px = particles_vis[:,1]
    py = particles_vis[:,2]
 
    # gen MET = (px, py) of genMET
    # uT = (-1)*genMET
    true_px = (-1)*genMET[:,0]
    true_py = (-1)*genMET[:,1]

    v_true = torch.stack((true_px,true_py),dim=1)
    true_uT = getscale(v_true)
    
    # regress uT: MET = (-1)*uT
    # ML weights are [0,1]
    uTx = scatter_add(weights*px, batch)
    uTy = scatter_add(weights*py, batch)

    loss=0.5*( ((uTx - true_px)**2 + (uTy - true_py)**2) / true_uT**2 ).mean() 

    return loss


# loss function flatten MET
def loss_fn_flattenMET(weights, particles_vis, genMET, batch, sample_weight = None):
    # particles_vis: (pT, px, py, eta, phi, puppiWeight, pdgId, charge)
    # momentum of the visible particles
    px = particles_vis[:,1]
    py = particles_vis[:,2]
 
    # gen MET = (px, py) of genMET
    # uT = (-1)*genMET
    true_px = (-1)*genMET[:,0]
    true_py = (-1)*genMET[:,1]

    # regress uT: MET = (-1)*uT
    # ML weights are [0,1]
    uTx = scatter_add(weights*px, batch)
    uTy = scatter_add(weights*py, batch)
    
    # flatten out MET
    if sample_weight != None:
        binnings = [0, 20, 40, 60, 80, 100, 120, 140, 160, 1000]

        per_genMET_bin_weight = [0.14890485, 0.05692836, 0.04364244, 0.04413395, 0.05471598, \
                                 0.08185655, 0.13920728, 0.25549501, 0.17511559]
        per_genMET_bin_weight = torch.tensor(per_genMET_bin_weight)

        v_true = torch.stack((true_px,true_py),dim=1)
        
        true_uT = getscale(v_true)

        for idx in range(len(binnings)-1):
            mask_uT = (true_uT > binnings[idx]) & (true_uT <= binnings[idx+1])

            sample_weight[mask_uT] = per_genMET_bin_weight[idx]

        #print(true_uT)
        #print(sample_weight)

        loss=0.5*( ( ( uTx - true_px)**2 + ( uTy - true_py)**2 ) * sample_weight ).mean()

    else:
        loss=0.5*( ( uTx - true_px)**2 + ( uTy - true_py)**2 ).mean()
        
    return loss



# calculate performance metrics
def metric(weights, particles_vis, genMET, batch, scale_momentum = 128.):
    # qT is the genMET
    qTx = genMET[:,0]
    qTy = genMET[:,1]

    v_qT=torch.stack((qTx,qTy),dim=1)

    # momentum of visible particles
    px = particles_vis[:,1]
    py = particles_vis[:,2]

    # regressed uT: momentum of the system of all visible particles
    uTx = scatter_add(weights*px, batch) 
    uTy = scatter_add(weights*py, batch) 
    
    # regressed MET
    METx = (-1) * uTx * scale_momentum
    METy = (-1) * uTy * scale_momentum

    v_MET=torch.stack((METx, METy),dim=1)

    # PUPPI MET using PUPPI weights
    wgt_puppi = particles_vis[:,5]

    puppiMETx = (-1)*scatter_add(wgt_puppi*px, batch) * scale_momentum
    puppiMETy = (-1)*scatter_add(wgt_puppi*py, batch) * scale_momentum
   
    v_puppiMET = torch.stack((puppiMETx, puppiMETy),dim=1)

    def compute(vector):
        response = getdot(vector,v_qT)/getdot(v_qT,v_qT)
        v_paral_predict = scalermul(response, v_qT)
        u_paral_predict = getscale(v_paral_predict)-getscale(v_qT)
        v_perp_predict = vector - v_paral_predict
        u_perp_predict = getscale(v_perp_predict)
        return [u_perp_predict.cpu().detach().numpy(), u_paral_predict.cpu().detach().numpy(), response.cpu().detach().numpy()]

    resolutions = {
        'MET':      compute(v_MET),
        'puppiMET': compute(v_puppiMET)
    }

    
    # gen MET, regressed MET, and PUPPI MET
    METs = {
        'genMETx': qTx.cpu().detach().numpy(),
        'genMETy': qTy.cpu().detach().numpy(),
        'genMET': getscale(v_qT).cpu().detach().numpy(),
        
        'METx': METx.cpu().detach().numpy(),
        'METy': METy.cpu().detach().numpy(),
        'MET': getscale(v_MET).cpu().detach().numpy(),
        
        'puppiMETx': puppiMETx.cpu().detach().numpy(),
        'puppiMETy': puppiMETy.cpu().detach().numpy(),
        'puppiMET': getscale(v_puppiMET).cpu().detach().numpy()
    }

    
    mask_down = torch.abs(particles_vis[:,6]) == 1
    mask_up = torch.abs(particles_vis[:,6]) == 2
    mask_electron = torch.abs(particles_vis[:,6]) == 11
    mask_muon = torch.abs(particles_vis[:,6]) == 13
    mask_photon = torch.abs(particles_vis[:,6]) == 22
    mask_kaon_zero = torch.abs(particles_vis[:,6]) == 130
    mask_pion_charged = torch.abs(particles_vis[:,6]) == 211
    
    weights = {
        'down': weights[mask_down].detach().cpu().numpy(),
        'up': weights[mask_up].detach().cpu().numpy(),
        'electron': weights[mask_electron].detach().cpu().numpy(),
        'muon': weights[mask_muon].detach().cpu().numpy(),
        'photon': weights[mask_photon].detach().cpu().numpy(),
        'kaon': weights[mask_kaon_zero].detach().cpu().numpy(),
        'pion': weights[mask_pion_charged].detach().cpu().numpy(),
        
    }
    
    puppi_weights = {
        'down': particles_vis[:,5][mask_down].detach().cpu().numpy(),
        'up': particles_vis[:,5][mask_up].detach().cpu().numpy(),
        'electron': particles_vis[:,5][mask_electron].detach().cpu().numpy(),
        'muon': particles_vis[:,5][mask_muon].detach().cpu().numpy(),
        'photon': particles_vis[:,5][mask_photon].detach().cpu().numpy(),
        'kaon': particles_vis[:,5][mask_kaon_zero].detach().cpu().numpy(),
        'pion': particles_vis[:,5][mask_pion_charged].detach().cpu().numpy(),
    }
    
    return resolutions, METs, weights, puppi_weights
@property
def losses(self):
    return [l.loss for l in self.layers if hasattr(l, 'loss')]

@property
def weights(self):
    return [l for l in self.layers if hasattr(l, 'constraint')]
# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'resolution': metric
}

