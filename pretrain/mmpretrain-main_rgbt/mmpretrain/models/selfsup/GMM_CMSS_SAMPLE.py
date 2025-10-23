import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

class GMM_CMSS_SAMPLE:
    def __init__(self):
        self.maskratio_max=0.95 # max mask ratio
        self.maskratio_bias = 0.0
        #epoch
        self.total_sample_num = 548238//2
        self.batch_size = 256*4
        self.count_flag = self.total_sample_num/self.batch_size
        #shift
        self.shift = 0.0
        self.sample_gmm_bias_max = 0.1
        self.sample_gmm_bias = 0.0
        #Gaussian mixture model
        self.n_components = 3
        self.weight = 1.0/self.n_components
        self.GMM_means = np.random.random(self.n_components)
        self.GMM_covariances = np.array([1.0 for i in range(self.n_components)])
        self.GMM_weights = np.array([self.weight for i in range(self.n_components)])
        self.sample_range = [0.0]
        self.GMM_weights_log = None
        self.GMM_means_log = None
        self.count = 0
        self.epoch = 0
        self.epoch_num = 100

    def e_step(self, X):
        self.resp = np.zeros((X.shape[0], self.n_components))
        for k in range(self.n_components):
            self.resp[:, k] = self.GMM_weights[k] * self.gaussian(X, self.GMM_means[k], self.GMM_covariances[k])
        self.resp = self.resp / self.resp.sum(axis=1, keepdims=True) #+0.00001)

    def m_step(self, X):
        N_k = self.resp.sum(axis=0)
        self.GMM_means = (self.resp.T @ X)/ N_k#[:,np.newaxis]
        self.GMM_covariances = np.array([
            np.sum(self.resp[:, k].T @ (X - self.GMM_means[k])**2) / N_k[k]#!!!!
            for k in range(self.n_components)
        ])
        self.GMM_weights = N_k / X.shape[0]

    def gaussian(self, X, mean, cov):
        return (1 / np.sqrt(2 * np.pi * cov)) * np.exp(-0.5 * ((X - mean)**2 / cov))


    def generate_GMM_PDF(self, N, L, x, sample_gmm_bias):

        GMM_means =  np.append(self.GMM_means,0)
        GMM_covariances =  np.append(self.GMM_covariances,0.01)
        GMM_weights =  np.append(self.GMM_weights,0.0)
        # GMM_means = np.append(GMM_means,1.0)
        # GMM_covariances = np.append(GMM_covariances,0.01)
        sort_indices = np.argsort(GMM_means)
        GMM_means_sort = GMM_means[sort_indices]# small->big
        GMM_covariances_sort = GMM_covariances[sort_indices]
        GMM_weights_sort = GMM_weights[sort_indices]

        # self.sample_range[1] = sample_var
        sample_GMM_weights = GMM_weights_sort[1:len(self.sample_range)+1]#
        sample_GMM_weights = sample_GMM_weights/np.sum(sample_GMM_weights)
        sample_list = []
        sample_count = 0
        for i in range(len(self.sample_range)):
            if i == len(self.sample_range)-1:
                sample_num = N*L-sample_count
            else:
                sample_num = int(N*L*sample_GMM_weights[i])
                sample_count = sample_count+sample_num
            mean = self.sample_range[i]+sample_gmm_bias#[len(self.sample_range)-1-i]
            sample_var = np.interp(mean, GMM_means_sort, GMM_covariances_sort)
            sample_x = torch.randn(sample_num, device=x.device) * np.sqrt(sample_var) + mean
            sample_list.append(sample_x)
        sample_all = torch.cat(sample_list, dim=0)
        sample_all = sample_all.view(N, L)
        self.GMM_weights_log =sample_GMM_weights
        self.GMM_means_log =  [mean, sample_var]
        # sample_x = torch.randn(N, L, device=x.device) * np.sqrt(sample_var) + mean
        return sample_all

    def MSDI_Similarity(self, tensor_1, tensor_2):
        normalized_tensor_1 = tensor_1 /tensor_1.norm(dim=-1,keepdim=True)
        normalized_tensor_2 = tensor_2 /tensor_2.norm(dim=-1,keepdim=True)
        r_num = torch.sum(normalized_tensor_1 * normalized_tensor_2, dim=1)
        r_num = (r_num+1.0)*0.5
        var_tensor_1 = torch.var(tensor_1, dim=1)
        var_tensor_2 = torch.var(tensor_1, dim=1)
        r_num = torch.sqrt(r_num)
        measure = r_num / (var_tensor_1*var_tensor_2)

        measure = measure / torch.max(measure)
        #measure = (measure - torch.min(measure)) / (torch.max(measure)-torch.min(measure))    
        return measure

    def cal_ids_shuffle(self,noise,cos_sim,cos_sim2,x,len_keep,first_modality=True):
        N, L, D = x.shape
        ids_shuffle = torch.zeros(N,L,device=x.device)
        for l in range(len_keep):
            abs_diff = torch.abs(noise[:, l].unsqueeze(1) - cos_sim[:, :])#B L
            nearest_index = torch.argsort(abs_diff)#[0]
            ids_shuffle[:,l] =nearest_index[:,0]
            cos_sim[range(N), nearest_index[:,0]] = 1e5 #nearest_index[:,0]-> 12
            if first_modality:
                cos_sim2[range(N), nearest_index[:,0]] = 1e3 #nearest_index[:,0]-> 12

        abs_diff = torch.abs(noise - cos_sim)
        nearest_index = torch.argsort(abs_diff)
        ids_shuffle[:, len_keep:] = nearest_index[:, :L-len_keep ]
        ids_shuffle = ids_shuffle.to(torch.int64)
        if first_modality:
            return ids_shuffle, cos_sim2
        else:
            return ids_shuffle

    def sample(self,x,x2,loss1,loss2,mask_ratio):
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        #cal the corr
        x_flat = x.clone().detach().contiguous().view(N*L,D)
        x2_flat = x2.clone().detach().contiguous().view(N*L,D)
        msdi_value = self.MSDI_Similarity(x_flat,x2_flat) #N*L

        # E step
        msdi_value_np = msdi_value.cpu().numpy()# N*L
        self.e_step(msdi_value_np)
        # M step
        self.m_step(msdi_value_np)

        # sample the mask area
        msdi_value  =msdi_value.view(N, L)# N L
        msdi_value2 = msdi_value.clone().detach()
        noise = self.generate_GMM_PDF(N, L, x, self.sample_gmm_bias)
        len_keep1 = len_keep-int(self.maskratio_bias)
        ids_shuffle,msdi_value2 = self.cal_ids_shuffle(noise, msdi_value, msdi_value2, x, len_keep1,first_modality=True)

        ids_restore = torch.argsort(ids_shuffle, dim=1)# 1.reorder  2.after reorder -> id
        ids_keep = ids_shuffle[:, :len_keep1]
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep1] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        #x2 mask RGB
        noise2 = self.generate_GMM_PDF(N, L, x2, 0-self.sample_gmm_bias)
        len_keep2 = len_keep+int(self.maskratio_bias)
        ids_shuffle2  = self.cal_ids_shuffle(noise2, msdi_value2, msdi_value2, x, len_keep2, first_modality=False)
        ids_restore2 = torch.argsort(ids_shuffle2, dim=1)  # 1.reorder  2.after reorder -> id
        ids_keep2 = ids_shuffle2[:, :len_keep2]
        x_masked2 = torch.gather(
            x2, dim=1, index=ids_keep2.unsqueeze(-1).repeat(1, 1, D))
        # generate the binary mask: 0 is keep, 1 is remove
        mask2 = torch.ones([N, L], device=x2.device)
        mask2[:, :len_keep2] = 0
        mask2 = torch.gather(mask2, dim=1, index=ids_restore2)

        #update the Gaussian  model
        self.count =  self.count +1
        if self.count> self.count_flag:
            self.count = 0
            self.epoch = self.epoch+1

            #delta
            delta_loss =  loss2 - loss1
            delta_loss = delta_loss.cpu().detach().numpy()
            if self.maskratio_bias < np.floor(L * (self.maskratio_max - mask_ratio)):
                self.maskratio_bias = self.maskratio_bias + delta_loss*10
            else:
                self.maskratio_bias = np.floor(L*(self.maskratio_max-mask_ratio))

            if self.sample_gmm_bias < self.sample_gmm_bias_max:
                self.sample_gmm_bias = self.sample_gmm_bias + delta_loss*0.1
            else:
                self.sample_gmm_bias = self.sample_gmm_bias_max

            if self.epoch<50:
                sort_indices = np.argsort(self.GMM_means)# small->big [::-1]
                GMM_means_sort = self.GMM_means[sort_indices]
                shift_flag = GMM_means_sort[len(self.sample_range)-1] - self.sample_range[-1]
                if shift_flag<0 and len(self.sample_range)<len(GMM_means_sort):
                    self.sample_range[-1] = GMM_means_sort[len(self.sample_range)-1] #v8.2
                    self.sample_range = self.sample_range + [0.0]
                self.shift = (np.sum(self.GMM_means) - np.sum(self.sample_range))/(50-self.epoch)
                self.sample_range[-1] = self.sample_range[-1] + self.shift
            else:
                sort_indices = np.argsort(self.GMM_means)# small->big [::-1]
                GMM_means_sort = self.GMM_means[sort_indices]
                self.sample_range[-1] = GMM_means_sort[-1]#0831!!!
                flag_j = 0
                for j in range(len(self.sample_range),0,-1):
                    if self.sample_range[j-1] < GMM_means_sort[-1]:
                        flag_j = j-1
                        break
                self.shift = (GMM_means_sort[-1]*len(GMM_means_sort) - np.sum(self.sample_range))/(101-self.epoch)
                self.sample_range[flag_j] = self.sample_range[flag_j] + self.shift


        return x_masked, mask, ids_restore, \
            x_masked2, mask2, ids_restore2
