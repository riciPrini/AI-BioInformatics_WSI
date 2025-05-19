import torch
from torch import nn
import torch.nn.functional as F
import math

import warnings
warnings.simplefilter("ignore", UserWarning)


class ABMIL(nn.Module):
    def __init__(self,
                     input_dim=1024,
                     inner_dim=64, 
                     output_dim=1, 
                     use_layernorm=True, 
                     dropout=0.0,
                ):
        super(ABMIL,self).__init__()

        self.inner_proj = nn.Linear(input_dim,inner_dim)
        self.output_dim = output_dim
        self.device="cuda" if torch.cuda.is_available() else "cpu"
        self.use_layernorm = use_layernorm
        self.dropout = nn.Dropout(dropout)
        if self.use_layernorm:
            self.layernorm = nn.LayerNorm(inner_dim)
        self.attention_V = nn.Linear(inner_dim, inner_dim)
        self.attention_U = nn.Linear(inner_dim, inner_dim)
        self.sigmoid = nn.Sigmoid()
        self.attention_weights = nn.Linear(inner_dim, 1)

        self.fc1 = nn.Linear(inner_dim, inner_dim//4)
        self.fc2 = nn.Linear(inner_dim//4, output_dim)
        #self.fc2 = nn.Linear(inner_dim, output_dim)
   
        
    def forward(self, data):
        x = self.inner_proj(data)
        
        if self.use_layernorm:
            x = self.layernorm(x)        
        
        # Apply attention mechanism
        V = torch.tanh(self.attention_V(x))  # Shape: (batch_size, num_patches, inner_dim)
        U = self.sigmoid(self.attention_U(x))  # Shape: (batch_size, num_patches, inner_dim)
        
        # Compute attention scores
        attn_scores = self.attention_weights(V * U)  # Shape: (batch_size, num_patches, 1)
        attn_scores = torch.softmax(attn_scores, dim=1)  # Shape: (batch_size, num_patches, 1)
        
        # Weighted sum of patch features
        weighted_sum = torch.sum(attn_scores * x, dim=1)  # Shape: (batch_size, inner_dim)
        weighted_sum = self.dropout(weighted_sum)

        # Final WSI embedding
        x = weighted_sum

        x = torch.relu(self.fc1(x)) # relu o tanh
        output = torch.sigmoid(self.fc2(x))
        
        return output, attn_scores
    

##########################################

class FCLayer(nn.Module):
    def __init__(self, in_size = 1024, out_size=1):
        super(FCLayer, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_size, out_size), nn.Sigmoid())
    def forward(self, feats):
        x = self.fc(feats)
        return feats, x

class BClassifier(nn.Module):#innerdim=128
    def __init__(self, input_size = 1024, output_dim = 1, inner_dim = 64, dropout_v=0.0, nonlinear=True, passing_v=True): # K, L, N
        super(BClassifier, self).__init__()
        if nonlinear:
            self.q = nn.Sequential(nn.Linear(input_size, inner_dim), nn.ReLU(), nn.Linear(inner_dim, inner_dim), nn.Tanh())
        else:
            self.q = nn.Linear(input_size, inner_dim)
        if passing_v:
            self.v = nn.Sequential(
                nn.Dropout(dropout_v),
                nn.Linear(input_size, inner_dim),
                nn.ReLU()
            )
        else:
            self.v = nn.Identity()
        
        self.fc1 = nn.Linear(inner_dim, inner_dim//4)
        self.fc2 = nn.Linear(inner_dim//4, output_dim)
    
        
    def forward(self, feats, c):
        device = feats.device
        V = self.v(feats)
        Q = self.q(feats)

        _, m_indices = torch.max(c, dim=1)


        critical_idx = m_indices.squeeze().item()#max prediction
        critical_patch = feats[0, critical_idx, :]
        critical_patch = critical_patch.unsqueeze(0).unsqueeze(0)
        q_max = self.q(critical_patch)

        # instance classifier
        instance_score = torch.sigmoid(c[:,critical_idx,:])

        ####
        A = torch.bmm(q_max, Q.transpose(1, 2))# controlla il transpose
        A = F.softmax( A / torch.sqrt(torch.tensor(Q.shape[2], dtype=torch.float32, device=device)), dim=-1) # questa sarà poi l'attention?
        B = torch.sum(A.transpose(1, 2)*V, dim = 1)#torch.bmm(A, V)

        x = torch.tanh(self.fc1(B)) # relu o tanh
        output = torch.sigmoid(self.fc2(x))

        return (output, instance_score), A
    
class MILNet(nn.Module):
    def __init__(self):
        super(MILNet, self).__init__()
        self.i_classifier = FCLayer()
        self.b_classifier = BClassifier()
        
    def forward(self, x):
        feats, classes = self.i_classifier(x)
        predictions, A = self.b_classifier(feats, classes)
        
        return predictions, A


#########################################################
####################### DS_ABMIL ########################

class ATTN_Score(nn.Module):
    def __init__(self,
                     input_dim=1024,
                     inner_dim=64,
                     output_dim=1, 
                     use_layernorm=False, 
                     dropout=0.0,
                ):
        super(ATTN_Score,self).__init__()

        self.inner_proj = nn.Linear(input_dim,inner_dim)
        self.output_dim = output_dim
        self.device="cuda" if torch.cuda.is_available() else "cpu"
        self.use_layernorm = use_layernorm
        self.dropout = nn.Dropout(dropout)
        if self.use_layernorm:
            self.layernorm = nn.LayerNorm(inner_dim)
        self.attention_V = nn.Linear(inner_dim, inner_dim)
        self.attention_U = nn.Linear(inner_dim, inner_dim)
        self.sigmoid = nn.Sigmoid()
        self.attention_weights = nn.Linear(inner_dim, 1)

   
        
    def forward(self, data):
        x = self.inner_proj(data)
        
        if self.use_layernorm:
            x = self.layernorm(x)        
        
        # Apply attention mechanism
        V = torch.tanh(self.attention_V(x))  # Shape: (batch_size, num_patches, inner_dim)
        U = self.sigmoid(self.attention_U(x))  # Shape: (batch_size, num_patches, inner_dim)
        
        # Compute attention scores
        attn_scores = self.attention_weights(V * U)  # Shape: (batch_size, num_patches, 1)
        attn_scores = torch.softmax(attn_scores, dim=1)  # Shape: (batch_size, num_patches, 1)

        return attn_scores

    
class DS_ABMIL(nn.Module):
    def __init__(self):
        super(DS_ABMIL, self).__init__()
        self.i_classifier = ATTN_Score(use_layernorm=True)
        self.b_classifier = BClassifier()
        
    def forward(self, x):
        scores = self.i_classifier(x)
        prediction_bag, A= self.b_classifier(x, scores)
        
        return prediction_bag, A



#######################################################################

if __name__ == '__main__':
    device = 'cuda'
    model =MILNet().to(device)
    model.eval()
    
    path = './TCGA-3C-AALK-01A-01-TSA.B64ED65E-C91A-42C9-89A5-1B099C7112C3.pt'
    data = torch.load(path, weights_only=True)
    data = data.unsqueeze(0)
    data = data.to(device)

    print(f'Memoria GPU prima: {torch.cuda.memory_allocated()/ (1024 ** 2)} Mbytes')
    
    out = model(data)

    print(f'Memoria GPU dopo: {torch.cuda.memory_allocated()/ (1024 ** 2)} Mbytes')