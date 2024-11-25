import torch
import torch.nn as nn
import torchvision.models as models

class CIFAR10MultivariateModel(nn.Module):
    def __init__(self):
        super(CIFAR10MultivariateModel, self).__init__()
        self.feature_extractor = models.resnet18(pretrained=False)
        self.feature_extractor.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.feature_extractor.maxpool = nn.Identity() 
        self.feature_extractor.fc = nn.Identity() 

        self.fc_mu = nn.Linear(512, 9)
        self.fc_r = nn.Linear(512, 9)

    def forward(self, x):
        features = self.feature_extractor(x)
        mu = self.fc_mu(features)
        r = self.fc_r(features) 
        return mu, r
    
class LabelCorrectionEMA:
    def __init__(self, model, device, decay=0.999):
        self.device = device
        self.model = model.to(device)
        self.decay = decay
        self.ema_model = self._clone_model()
        self.alpha = 0.99

    def _clone_model(self):
        ema_model = type(self.model)()
        ema_model.load_state_dict(self.model.state_dict())
        ema_model.to(self.device)
        for param in ema_model.parameters():
            param.detach_()
        return ema_model

    def update(self):
        for ema_param, model_param in zip(self.ema_model.parameters(), self.model.parameters()):
            ema_param.data = self.decay * ema_param.data + (1.0 - self.decay) * model_param.data

    def compute_shift(self, epoch, inputs, ilr_labels, mu):
        transition_weight = 1 - (self.alpha ** epoch)
        
        with torch.no_grad():
            ema_mu, _ = self.ema_model(inputs)
        
        shift = ilr_labels - ema_mu
        corrected_mu = mu + transition_weight * shift
        
        return corrected_mu

def construct_covariance_and_inverse(r):
    batch_size, rank = r.shape
    I = torch.eye(rank).to(r.device).unsqueeze(0).expand(batch_size, -1, -1)
    r = r.unsqueeze(2)
    outer_r = torch.bmm(r, r.transpose(1, 2))
    S = outer_r + I
    r_dot_r = torch.bmm(r.transpose(1, 2), r)
    S_inv = I - outer_r / (1 + r_dot_r)
    log_det_S = torch.log(1 + r_dot_r.squeeze(-1).squeeze(-1))
    return S_inv, log_det_S

def enhanced_multivariate_gaussian_loss(y_true, mu, r, shift=None):
    S_inv, log_det_S = construct_covariance_and_inverse(r)
    
    if shift is not None:
        mu = mu + shift
    
    diff = y_true - mu 
    mahalanobis_dist = torch.bmm(diff.unsqueeze(1), S_inv) 
    mahalanobis_dist = torch.bmm(mahalanobis_dist, diff.unsqueeze(2)).squeeze() 
    
    loss = 0.5 * mahalanobis_dist + 0.5 * log_det_S
    return loss.mean()
