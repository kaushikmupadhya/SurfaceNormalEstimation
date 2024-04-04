import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

#################################MARCH
class WrappedNormalLoss(nn.Module):
    def __init__(self):
        super(WrappedNormalLoss, self).__init__()

    def forward(self, pred_norm, gt_norm, gt_norm_mask):
        pred_norm, pred_kappa = pred_norm[:, 0:3, :, :], pred_norm[:, 3:, :, :]
        dot = torch.cosine_similarity(pred_norm, gt_norm, dim=1)
        valid_mask = gt_norm_mask[:, 0, :, :].float() * (dot.detach() < 0.999).float() * (dot.detach() > -0.999).float()
        valid_mask = valid_mask > 0.0
        dot = dot[valid_mask]
        kappa = pred_kappa[:, 0, :, :][valid_mask]
        loss_pixelwise = - torch.log(torch.square(kappa) + 1) + kappa * torch.acos(dot) + torch.log(1 + torch.exp(-kappa * np.pi))
        loss = torch.mean(loss_pixelwise)
        return loss 
##############################################

# compute loss
class compute_loss(nn.Module):
    def __init__(self, args):
        """args.loss_fn can be one of following:
            - L1            - L1 loss (no uncertainty)
            - L2            - L2 loss (no uncertainty)
            - AL            - Angular loss (no uncertainty)
            - NLL_vMF       - NLL of vonMF distribution
            - NLL_ours      - NLL of Angular vonMF distribution
            - WNLL          - NLL of Wrapped Normal distribution    #KSK
            - FWNL         - FEB 2024VERSION
            - WCLL          - NLL of Wrapped Cauchy distribution    #KSK
            - UG_NLL_vMF    - NLL of vonMF distribution (+ pixel-wise MLP + uncertainty-guided sampling)
            - UG_NLL_ours   - NLL of Angular vonMF distribution (+ pixel-wise MLP + uncertainty-guided sampling)
        """
        super(compute_loss, self).__init__()
        self.loss_type = args.loss_fn
        if self.loss_type in ['L1', 'L2', 'AL', 'NLL_vMF', 'NLL_ours', 'WP_NL']:
            self.loss_fn = self.forward_R
        elif self.loss_type in ['UG_NLL_vMF', 'UG_NLL_ours', 'WNLL', 'FWNL', 'WCLL']:  #KSK ('WNLL', 'WCLL')
            self.loss_fn = self.forward_UG
        else:
            raise Exception('invalid loss type')

    def forward(self, *args):
        return self.loss_fn(*args)

    def forward_R(self, norm_out, gt_norm, gt_norm_mask):
        pred_norm, pred_kappa = norm_out[:, 0:3, :, :], norm_out[:, 3:, :, :]

        if self.loss_type == 'L1':
            l1 = torch.sum(torch.abs(gt_norm - pred_norm), dim=1, keepdim=True)
            loss = torch.mean(l1[gt_norm_mask])

        elif self.loss_type == 'L2':
            l2 = torch.sum(torch.square(gt_norm - pred_norm), dim=1, keepdim=True)
            loss = torch.mean(l2[gt_norm_mask])
        ############MARCH
        elif self.loss_type == 'WP_NL':
            return self.wrapped_normal_loss(norm_out, gt_norm, gt_norm_mask)
        #############################
        elif self.loss_type == 'AL':
            dot = torch.cosine_similarity(pred_norm, gt_norm, dim=1)

            valid_mask = gt_norm_mask[:, 0, :, :].float() \
                         * (dot.detach() < 0.999).float() \
                         * (dot.detach() > -0.999).float()
            valid_mask = valid_mask > 0.0

            al = torch.acos(dot[valid_mask])
            loss = torch.mean(al)

        elif self.loss_type == 'NLL_vMF':
            dot = torch.cosine_similarity(pred_norm, gt_norm, dim=1)

            valid_mask = gt_norm_mask[:, 0, :, :].float() \
                         * (dot.detach() < 0.999).float() \
                         * (dot.detach() > -0.999).float()
            valid_mask = valid_mask > 0.0

            dot = dot[valid_mask]
            kappa = pred_kappa[:, 0, :, :][valid_mask]

            loss_pixelwise = - torch.log(kappa) \
                             - (kappa * (dot - 1)) \
                             + torch.log(1 - torch.exp(- 2 * kappa))
            loss = torch.mean(loss_pixelwise)

        elif self.loss_type == 'NLL_ours':
            dot = torch.cosine_similarity(pred_norm, gt_norm, dim=1)

            valid_mask = gt_norm_mask[:, 0, :, :].float() \
                         * (dot.detach() < 0.999).float() \
                         * (dot.detach() > -0.999).float()
            valid_mask = valid_mask > 0.0

            dot = dot[valid_mask]
            kappa = pred_kappa[:, 0, :, :][valid_mask]

            loss_pixelwise = - torch.log(torch.square(kappa) + 1) \
                             + kappa * torch.acos(dot) \
                             + torch.log(1 + torch.exp(-kappa * np.pi))
            loss = torch.mean(loss_pixelwise)

        else:
            raise Exception('invalid loss type')

        return loss


    def forward_UG(self, pred_list, coord_list, gt_norm, gt_norm_mask):
        loss = 0.0
        for (pred, coord) in zip(pred_list, coord_list):
            if coord is None:

                

                pred = F.interpolate(pred, size=[gt_norm.size(2), gt_norm.size(3)], mode='bilinear', align_corners=True)
                pred_norm, pred_kappa = pred[:, 0:3, :, :], pred[:, 3:, :, :]
                
                print("\n## Loss type:", self.loss_type)
                print("NO Sampling -INSIDE forward_UG coord is None ")###########KSK
                print("NO Sampling Shape pred_norm:", pred_norm.shape)###########KSK
                print("NO Sampling Shape pred_kappa:", pred_kappa.shape)   ###########KSK
                print("NO Sampling Shape gt_norm:", gt_norm.shape)###########KSK




                if self.loss_type == 'UG_NLL_vMF':
                    dot = torch.cosine_similarity(pred_norm, gt_norm, dim=1)

                    valid_mask = gt_norm_mask[:, 0, :, :].float() \
                                * (dot.detach() < 0.999).float() \
                                * (dot.detach() > -0.999).float()
                    valid_mask = valid_mask > 0.5

                    # mask
                    dot = dot[valid_mask]
                    kappa = pred_kappa[:, 0, :, :][valid_mask]

                    loss_pixelwise = - torch.log(kappa) \
                                     - (kappa * (dot - 1)) \
                                     + torch.log(1 - torch.exp(- 2 * kappa))
                    loss = loss + torch.mean(loss_pixelwise)

                elif self.loss_type == 'UG_NLL_ours':
                    dot = torch.cosine_similarity(pred_norm, gt_norm, dim=1)

                    valid_mask = gt_norm_mask[:, 0, :, :].float() \
                                * (dot.detach() < 0.999).float() \
                                * (dot.detach() > -0.999).float()
                    valid_mask = valid_mask > 0.5

                    dot = dot[valid_mask]
                    kappa = pred_kappa[:, 0, :, :][valid_mask]

                    loss_pixelwise = - torch.log(torch.square(kappa) + 1) \
                                     + kappa * torch.acos(dot) \
                                     + torch.log(1 + torch.exp(-kappa * np.pi))
                    loss = loss + torch.mean(loss_pixelwise)

                ####ADDED BY KSK######################
                elif self.loss_type == 'WNLL':  ################ non sampling wnll###############
                    # pred_kappa = pred[:, 3:]
##################### FEB 7
                    pred_norm, pred_kappa = pred[:, 0:3, :, :], pred[:, 3:, :, :]
                    # pred_norm_wnll = F.interpolate(pred_norm, size=[gt_norm.size(2), gt_norm.size(3)], mode='bilinear', align_corners=True)
                    # pred_kappa_wnll = F.interpolate(pred_kappa, size=[gt_norm.size(2), gt_norm.size(3)], mode='bilinear', align_corners=True)
                    # pred_kappa_wnll = pred_kappa_wnll[:, 0, :, :]  # Remove the extra channel dimension

                    # pred_norm_wnll = pred_norm_wnll.permute(0, 2, 3, 1)  # (B, H, W, 3)
                    
                    gt_norm_wnll = F.interpolate(gt_norm.permute(0, 2, 3, 1), size=pred_norm.shape[1:3], mode='bilinear', align_corners=True).contiguous()
                    # gt_norm_wnll = gt_norm_wnll.view(gt_norm_wnll.shape[0], gt_norm_wnll.shape[1], gt_norm_wnll.shape[2], 3)
                    # gt_norm_wnll = F.interpolate(gt_norm.permute(0, 2, 3, 1), size=pred_norm_wnll.shape[1:3], mode='bilinear', align_corners=True)
                    ###################################################

                    print("Inside the elif WNLL loss") 
                    print("\npred_kappa shape:", pred_kappa.shape)  
                    print("\npred_norm shape:", pred_norm.shape)  
                    print("\ngt_norm shape:", gt_norm_wnll.shape)  
                    # Wrapped normal loss
                    # loss_wnll = self.wrapped_normal_loss(pred_norm_wnll, pred_kappa_wnll, gt_norm_wnll)
                    loss_wnll = self.wrapped_normal_loss(pred_norm, pred_kappa, gt_norm_wnll)
                    # print("Loss WNLL:", loss_wnll)######KSK
                    loss += loss_wnll

                elif self.loss_type == 'FWNL': #### NON SAMPLING FEB VERSION
                    gt_norm_ = gt_norm
                    pred_norm = pred[:, 0:3] #shapes of gt_norm_ and pred_norm should match
                    pred_kappa = pred[:, 3:] #added on march 3
                    dot = torch.cosine_similarity(pred_norm, gt_norm, dim=1)
                    dot = torch.clamp(dot, min=-1.0, max=1.0)  
                    
                    print("\nInside the elif FWNL loss - NO Sampling") 
                    print("FWNL NO Sampling pred_norm shape:", pred_norm.shape)  
                    print("FWNL NO Sampling pred_kappa shape:", pred_kappa.shape)  
                    print("FWNL NO Sampling gt_norm shape:", gt_norm_.shape)     
                    
                    theta = torch.acos(dot)
                    kappa = pred_kappa[:, 0]
                    
                    # Rest same as sampled loss
                    b = 1 / (1 + torch.exp(-kappa)) 
                    a = kappa * b
                    
                    const = torch.tensor(0.5 * np.log(2 * np.pi))
                    loss_pixelwise = const - torch.log(b) \
                                    + 0.5 * (a * theta) ** 2 \
                                    + 0.5 * torch.log(1 + 1 / (b * 2))
                    
                    loss += torch.mean(loss_pixelwise)

        
                elif self.loss_type == 'WCLL':
                    pred_gamma = pred[:, 3:]
                
                    # Wrapped cauchy loss
                    loss_wcll = self.wrapped_cauchy_loss(pred_norm, pred_gamma, gt_norm)
                    loss += loss_wcll
                #######################################

                else:
                    raise Exception

            else:  ############### SAMPLING PART########################
                # coord: B, 1, N, 2
                # pred: B, 4, N

                print("Shape of coord:", coord.shape) ##FEB20

                gt_norm_ = F.grid_sample(gt_norm, coord, mode='nearest', align_corners=True)  # (B, 3, 1, N)

                

                gt_norm_mask_ = F.grid_sample(gt_norm_mask.float(), coord, mode='nearest', align_corners=True)  # (B, 1, 1, N)
                gt_norm_ = gt_norm_[:, :, 0, :]  # (B, 3, N)
                gt_norm_mask_ = gt_norm_mask_[:, :, 0, :] > 0.5  # (B, 1, N)

                pred_norm, pred_kappa = pred[:, 0:3, :], pred[:, 3:, :]
                print("\n## Loss type:", self.loss_type)
                print("SAMPLING - INSIDE forward_UG coord ")###########KSK   
                print("Sampling Shape pred_norm:", pred_norm.shape)###########KSK
                print("Sampling Shape pred_kappa:", pred_kappa.shape)   ###########KSK
                print("Sampling Shape gt_norm:", gt_norm.shape)###########KSK
                ###################KSK
                # if pred_norm.shape != gt_norm.shape:
                #     print("Shape mismatch!")
                #     return
                # if pred_kappa.sum() == 0.:
                #     print("Invalid pred_kappa")
                #     return
                ####################

                if self.loss_type == 'UG_NLL_vMF':
                    dot = torch.cosine_similarity(pred_norm, gt_norm_, dim=1)  # (B, N)

                    valid_mask = gt_norm_mask_[:, 0, :].float() \
                                 * (dot.detach() < 0.999).float() \
                                 * (dot.detach() > -0.999).float()
                    valid_mask = valid_mask > 0.5

                    dot = dot[valid_mask]
                    kappa = pred_kappa[:, 0, :][valid_mask]

                    loss_pixelwise = - torch.log(kappa) \
                                     - (kappa * (dot - 1)) \
                                     + torch.log(1 - torch.exp(- 2 * kappa))
                    loss = loss + torch.mean(loss_pixelwise)

                elif self.loss_type == 'UG_NLL_ours':
                    dot = torch.cosine_similarity(pred_norm, gt_norm_, dim=1)  # (B, N)

                    valid_mask = gt_norm_mask_[:, 0, :].float() \
                                 * (dot.detach() < 0.999).float() \
                                 * (dot.detach() > -0.999).float()
                    valid_mask = valid_mask > 0.5

                    dot = dot[valid_mask]
                    kappa = pred_kappa[:, 0, :][valid_mask]

                    loss_pixelwise = - torch.log(torch.square(kappa) + 1) \
                                     + kappa * torch.acos(dot) \
                                     + torch.log(1 + torch.exp(-kappa * np.pi))
                    loss = loss + torch.mean(loss_pixelwise)

                ####ADDED BY KSK######################
                elif self.loss_type == 'WNLL':  ############### SAMPLING W N L L ########################
                    # pred_kappa = pred[:, 3:]
                    print("Compute WNLL loss") 
                    gt_norm_ = F.grid_sample(gt_norm, coord, mode='nearest')

# ############ADDED 7Feb
                    gt_norm_ = gt_norm_.squeeze(2)  # remove singleton dim
                    gt_norm_ = gt_norm_.transpose(1, 2)  # make channels dim 2

                    if coord_list and coord_list[0] is not None:
                        pred_mlp, pred_coord = pred_list[0], coord_list[0]

                        pred_norm_wnll = F.grid_sample(pred_mlp[:, 0:3], pred_coord, mode='bilinear', align_corners=True)
                        pred_norm_wnll = pred_norm_wnll[:, :, 0, :]  # (B, 3, N)
                        pred_kappa_wnll = F.grid_sample(pred_mlp[:, 3:], pred_coord, mode='bilinear', align_corners=True)
                        pred_kappa_wnll = pred_kappa_wnll[:, :, 0, :]  # (B, 1, N)
                    else:
                        pred_norm_wnll = pred_norm
                        pred_kappa_wnll = pred_kappa

                    # pred_mlp, pred_coord = pred_list[0], coord_list[0]

                    print("Inside the elif WNLL loss") 
                    print("\npred_kappa shape:", pred_kappa_wnll.shape)  
                    print("\npred_norm shape:", pred_norm_wnll.shape)  
                    print("\ngt_norm shape:", gt_norm_.shape)  

                    # pred_norm_ = F.grid_sample(pred_mlp[:, 0:3], pred_coord, mode='bilinear', align_corners=True)
                    # pred_norm_ = pred_norm_[:, :, 0, :]  # (B, 3, N)
                    # pred_kappa_ = F.grid_sample(pred_mlp[:, 3:], pred_coord, mode='bilinear', align_corners=True)
                    # pred_kappa_ = pred_kappa_[:, :, 0, :]  # (B, 1, N)
                     # Wrapped normal loss
                    loss_wnll = self.wrapped_normal_loss(pred_norm_wnll, pred_kappa_wnll, gt_norm_)
                # print("Loss WNLL:", loss_wnll)
                    loss += loss_wnll
# ############ADDED 7Feb
                    
#                     pred_norm, pred_kappa = pred[:, 0:3], pred[:, 3:] # MLP predictions

#                     print("Inside the elif WNLL loss") 
#                     print("\npred_kappa shape:", pred_kappa.shape)  
#                     print("\npred_norm shape:", pred_norm.shape)  
#                     print("\ngt_norm shape:", gt_norm.shape)  
                    
#                     # Reshape pred_norm and pred_kappa to match the batch dimension of gt_norm_
#                     pred_norm = pred_norm.view(gt_norm_.shape[0], 3, -1)
#                     pred_kappa = pred_kappa.view(gt_norm_.shape[0], -1)
# # ############ADDED 7Feb
#                     print("IAfter reshaping") 
#                     print("\npred_kappa shape:", pred_kappa.shape)  
#                     print("\npred_norm shape:", pred_norm.shape)  
#                     print("\ngt_norm shape:", gt_norm.shape)  
# #                     gt_norm_ = F.grid_sample(gt_norm, coord, mode='nearest')  
# #                     gt_norm_ = gt_norm_.squeeze(2) # remove singleton dim
# #                     gt_norm_ = gt_norm_.transpose(1,2) # make channels dim 2

# #                     pred_norm, pred_kappa = pred[:, 0:3], pred[:, 3:] 

# #############################                                        
                
#                     # Wrapped normal loss
#                     loss_wnll = self.wrapped_normal_loss(pred_norm, pred_kappa, gt_norm)
#                     # print("Loss WNLL:", loss_wnll)######KSK
#                     loss += loss_wnll
                    
                elif self.loss_type == 'FWNL': ###FEB VERSION SAMPLED LOSS
                    # # gt_norm_ = F.grid_sample(gt_norm, coord, mode='nearest')    
                    # print("Shape of gt_norm_ after grid_sample:", gt_norm_.shape) ##FEB20
                    # num_points = torch.numel(gt_norm_) // gt_norm.size(1)##FEB20
                    # print(num_points)##FEB20
                    # # Reshape gt_norm_ to match pred shape
                    # # gt_norm_ = gt_norm_.view(gt_norm.size(0), gt_norm.size(1), -1)  
                    # gt_norm_ = gt_norm_.view(gt_norm.size(0), gt_norm.size(1), num_points)
                    # # gt_norm_ = gt_norm_.unsqueeze(2)

                    # gt_norm_ = F.grid_sample(gt_norm, coord)
                    print("###ADDED BY KSK######################")

                    pred_norm = torch.clamp(pred_norm, -1, 1) #March 1
                    gt_norm_ = F.grid_sample(gt_norm, coord)  #March 1

                    print(gt_norm_.shape)
                    N = gt_norm_.size(-1)
                    print(N)
                    
                    gt_norm_ = gt_norm_.view(gt_norm.size(0), gt_norm.size(1), N)
                    gt_norm_ = gt_norm_.squeeze(2) #March 1
                    print(gt_norm_.shape)
                    print("###ADDED BY KSK######################")
                    # gt_norm_ = gt_norm_.unsqueeze(2)
                    
                    print("\n\nInside the elif FWNL loss - Sampling") 
                    print("FWNL Sampling pred_norm shape:", pred_norm.shape) 
                    print("FWNL Sampling pred_kappa shape:", pred_kappa.shape)  
                    print("FWNL Sampling gt_norm shape:", gt_norm_.shape) 

                    dot = torch.cosine_similarity(pred_norm, gt_norm_, dim=1)
                    dot = torch.clamp(dot, min=-1.0, max=1.0)
                    
                    theta = torch.acos(dot)  
                    pred_kappa = torch.clamp(pred_kappa, 0, 10) #march1
                    kappa = pred_kappa[:, 0]
                    
                    b = 1 / (1 + torch.exp(-kappa))
                    a = kappa * b
                    
                    const = torch.tensor(0.5 * np.log(2 * np.pi))
                    loss_pixelwise = const - torch.log(b) \
                                    + 0.5 * (a * theta) ** 2 \
                                    + 0.5 * torch.log(1 + 1 / (b * 2))
                                    
                    loss += torch.mean(loss_pixelwise)

                elif self.loss_type == 'WCLL':
                    pred_gamma = pred[:, 3:]
                    
                    # Wrapped cauchy loss
                    loss_wcll = self.wrapped_cauchy_loss(pred_norm, pred_gamma, gt_norm)
                    loss += loss_wcll
                #######################################

                else:
                    raise Exception
        return loss
    

    ###############ADDED BY KSK###########################
    def wrapped_normal_loss(self, pred_norm, pred_kappa, gt_norm):
        """
        Compute wrapped normal loss between predicted and ground truth normals.

        Args:
        pred_norm: Predicted normal vector (B x 3 x H x W)
        pred_kappa: Predicted concentration (B x 1 x H x W)  
        gt_norm: Ground truth normal vector (B x 3 x H x W)
        coord: Sample coordinates  
        Returns:
        loss_wnll: Wrapped normal NLL loss  
        """
         # Ensure operations are performed with tensors
        pi_tensor = torch.tensor(np.pi, dtype=pred_norm.dtype, device=pred_norm.device)  # Convert pi to a tensor
       
        ##########FEB7
        # Transpose gt_norm to match the shape of pred_norm
        # gt_norm = gt_norm.transpose(1, 2)  # (B x N x 3)

        ############################



        # Calculate angle 
        dot = torch.cosine_similarity(pred_norm, gt_norm, dim=1)  # dot: (B x N)
        dot = torch.clamp(dot, min=1.0, max=1.0)  # Ensure dot values are within the valid range for acos
        theta = torch.acos(dot) # theta: (B x N)

        #############feb7
        # theta = theta.mean(dim=1, keepdim=True)

        #########################

        
        # Wrapped normal distribution  
        # Source: Mardia (2009)
        b = 1 / (1 + torch.exp(-pred_kappa))
        a = pred_kappa * b
        
        print("Inside wrapped normal loss function")
        print("Shape pred_norm:", pred_norm.shape)
        print("Shape pred_kappa:", pred_kappa.shape)
        print("a shape:", a.shape)
        print("theta shape:", theta.shape)
        # Negative log likelihood loss
        # Source: Mardia (2009)
        loss_wnll = 0.5*torch.log(2 * pi_tensor) - torch.log(b) \
                   + 0.5 * (a * theta)**2 \
                   + 0.5 * torch.log(1 + 1/(b**2))
        print("\n\nWNLL shape:", loss_wnll.shape)           
        return loss_wnll

    # Calculation as before

    def wrapped_cauchy_loss(pred_norm, pred_gamma, gt_norm):
        """
        Compute wrapped Cauchy loss between predicted and ground truth normals.

        Args:  
        pred_norm: Predicted normal vector (B x 3 x H x W)
        pred_gamma: Predicted concentration (B x 1 x H x W)
        gt_norm: Ground truth normal vector (B x 3 x H x W)
        coord: Sample coordinates
    
        Returns:  
        loss_wcll: Wrapped Cauchy NLL loss 
        """
        
        # Calculate angle   
        dot = torch.cosine_similarity(pred_norm, gt_norm, dim= 2) 
        theta = torch.acos(dot)
        # Wrapped Cauchy distribution 
        # Negative log likelihood loss
        # Source: Kurz (2013) 
        loss_wcll = torch.log(np.pi) - torch.log(pred_gamma) \
                   + torch.log(1 + (theta / pred_gamma)**2)
        return loss_wcll
##################################################33