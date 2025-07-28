import torch
import numpy as np

from .core.NonstationaryGpModule import NonstationaryGpModule
from .utils.utils import normalize_coordinates

def sample_coords(depth, num_samples):
    B = depth.shape[0]
    H = depth.shape[-2]
    W = depth.shape[-1]
    device = depth.device
    y_coords, x_coords = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    coord_img = torch.dstack((y_coords, x_coords)).unsqueeze(0).repeat(B,1,1,1)

    depth_vec = torch.reshape(depth, (B,-1))
    coord_vec = torch.reshape(coord_img, (B,-1,2))

    valid_depths = ~torch.isnan(depth_vec)

    weights = 1.0*valid_depths

    inds = torch.multinomial(weights, num_samples, replacement=False)
    batch_inds = torch.arange(B,device=device).unsqueeze(1).repeat(1,num_samples)
    coord_samples = coord_vec[batch_inds, inds, :] 
    depth_samples = depth_vec[batch_inds, inds]
    depth_samples = depth_samples.unsqueeze(-1)

    return coord_samples, depth_samples, batch_inds


class DepthCovEstimator:
    
    def __init__(self, config):
        
        self.level = config['level']
        assert self.level == -1, "Need to adjust resize in query if going to use other levels"
        self.model_path = config['model_path']
        
        self.model = NonstationaryGpModule.load_from_checkpoint(self.model_path)
        self.model.eval()
        
        self.device = config['device']
        self.model.to(self.device)
        
        self.std_valid_threshold = config['std_valid_threshold']
        
    def norm_coords(self, coords, image_size):
        # First axis is the height, y axis
        return normalize_coordinates(coords, image_size)
    
    def query(self, rgb_orig, sparse_depth, train_coords_xy_np, test_coords_xy_np):
        with torch.no_grad():    
            rgb = rgb_orig.permute(2, 0, 1).unsqueeze(0)
            rgb = rgb.to(self.device)
            
            img_size = rgb.shape[-2:]
            
            train_coords_xy = torch.from_numpy(train_coords_xy_np)
            train_coords = torch.zeros_like(train_coords_xy)
            train_coords[..., 0] = train_coords_xy[..., 1]
            train_coords[..., 1] = train_coords_xy[..., 0]
            train_coords = train_coords.to(self.device)
            train_coords = self.norm_coords(train_coords, img_size)
            train_coords = train_coords.unsqueeze(0).float()
            
            test_coords_xy = torch.from_numpy(test_coords_xy_np)
            test_coords = torch.zeros_like(test_coords_xy)
            test_coords[..., 0] = test_coords_xy[..., 1]
            test_coords[..., 1] = test_coords_xy[..., 0]
            test_coords = test_coords.to(self.device)
            test_coords = self.norm_coords(test_coords, img_size)
            test_coords = test_coords.unsqueeze(0).float()
            
            # target_size = 400*400
            
            # new_w, new_h = int(img_size[1] * np.sqrt(target_size / (img_size[0] * img_size[1]))), int(img_size[0] * np.sqrt(target_size / (img_size[0] * img_size[1])))
            # new_w, new_h = new_w // 2**5 * 2**5, new_h // 2**5 * 2**5
            
            new_w, new_h = img_size[1] // 2**5 * 2**5, img_size[0] // 2**5 * 2**5

            rgb = torch.nn.functional.interpolate(rgb, size=(new_w, new_h), mode='bilinear', align_corners=True)
            
            gaussian_covs = self.model(rgb)
            
            gaussian_covs[self.level] = torch.nn.functional.interpolate(gaussian_covs[self.level], size=(img_size[1], img_size[0]), mode='bilinear', align_corners=True)
            
            sparse_log_depth = torch.log(torch.from_numpy(sparse_depth).to(self.device))    
            sparse_log_depth = sparse_log_depth.unsqueeze(0).unsqueeze(2)
            
            mean_depth = torch.nanmean(sparse_log_depth)
            
            pred_depth_mean, pred_depth_var = self.model.est_sparse_depth(gaussian_covs, train_coords, sparse_log_depth, mean_depth, test_coords, level=self.level)
            
            return torch.exp(pred_depth_mean[0, 0]).detach().cpu().numpy(), (torch.sqrt(pred_depth_var[0, 0]) < self.std_valid_threshold).detach().cpu().numpy()
        
    def query_tensor(self, rgb_orig, sparse_depth, train_coords_xy, test_coords_xy):
        with torch.no_grad():    
            
            rgb = rgb_orig.permute(2, 0, 1).unsqueeze(0)
            rgb = rgb.to(self.device)
            
            img_size = rgb.shape[-2:]
            
            train_coords = torch.zeros_like(train_coords_xy).to(train_coords_xy.device)
            train_coords[..., 0] = train_coords_xy[..., 1]
            train_coords[..., 1] = train_coords_xy[..., 0]
            train_coords = train_coords.to(self.device)
            train_coords = self.norm_coords(train_coords, img_size)
            train_coords = train_coords.unsqueeze(0).float()
            
            test_coords = torch.zeros_like(test_coords_xy).to(test_coords_xy.device)
            test_coords[..., 0] = test_coords_xy[..., 1]
            test_coords[..., 1] = test_coords_xy[..., 0]
            test_coords = test_coords.to(self.device)
            test_coords = self.norm_coords(test_coords, img_size)
            test_coords = test_coords.unsqueeze(0).float()
            
            
            # target_size = 400*400
            
            # new_w, new_h = int(img_size[1] * np.sqrt(target_size / (img_size[0] * img_size[1]))), int(img_size[0] * np.sqrt(target_size / (img_size[0] * img_size[1])))
            # new_w, new_h = new_w // 2**5 * 2**5, new_h // 2**5 * 2**5
            
            new_w, new_h = img_size[1] // 2**5 * 2**5, img_size[0] // 2**5 * 2**5

            rgb = torch.nn.functional.interpolate(rgb, size=(new_w, new_h), mode='bilinear', align_corners=True)

            
            gaussian_covs = self.model(rgb)
    
            
            gaussian_covs[self.level] = torch.nn.functional.interpolate(gaussian_covs[self.level], size=(img_size[1], img_size[0]), mode='bilinear', align_corners=True)

            
            sparse_log_depth = torch.log(sparse_depth)    
            sparse_log_depth = sparse_log_depth.unsqueeze(0).unsqueeze(2)
            
            mean_depth = torch.nanmean(sparse_log_depth)
            
            pred_depth_mean, pred_depth_var = self.model.est_sparse_depth(gaussian_covs, train_coords, sparse_log_depth, mean_depth, test_coords, level=self.level)
            
            return torch.exp(pred_depth_mean[0, 0]), (torch.sqrt(pred_depth_var[0, 0]) < self.std_valid_threshold)
        
    def estimate_depth(self, rgbs, depths, sample_num = 1000):
        with torch.no_grad():
            
            img_size = rgbs.shape[-2:]
            B = rgbs.shape[0]
            
            depths_log = torch.log(depths)
            
            min_valid_depth_num = torch.min(torch.sum(depths.reshape(B, -1) > 0, dim=1))    
            sample_num = min(sample_num, min_valid_depth_num)
            
            train_coords_xy, train_depths, train_batch_inds = sample_coords(depths_log, sample_num)
            
            
            gaussian_covs = self.model(rgbs)
            
            coords_train_norm = self.norm_coords(train_coords_xy, img_size)
            mean_depth = torch.nanmean(train_depths)
            
            pred_depths, pred_vars = self.model.condition(gaussian_covs, coords_train_norm, train_depths, mean_depth, img_size)
            
            return torch.exp(pred_depths[-1]), pred_vars[-1]
            
            
            
            
            
            
            
            
    