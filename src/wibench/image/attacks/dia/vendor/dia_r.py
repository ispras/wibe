
import sys
import torch
from torchvision import transforms as tf
from PIL import Image
from tqdm import tqdm

TARGET_SIZE = 512
to_pil = tf.ToPILImage()
to_tensor = tf.ToTensor()
tform = tf.Compose(
    [
        tf.Resize(TARGET_SIZE),
        tf.CenterCrop(TARGET_SIZE),
        tf.ToTensor(), 
    ]
)
Image_to_tensor = lambda x: 2.0*tform(x)-1.0 
path_to_tensor = lambda x: Image_to_tensor(Image.open(x).convert('RGB')) 
is_vae_latent = lambda x: x.shape[1]!=3 
def backward_ddim(x_t, alpha_t: "alpha_t", alpha_tm1: "alpha_{t-1}", eps_xt):
    return ((alpha_tm1**0.5*alpha_t**-0.5)*x_t + alpha_tm1**0.5*((1 / alpha_tm1 - 1) ** 0.5 - (1 / alpha_t - 1) ** 0.5) * eps_xt)

def L1_projection(x2, y2, eps1):
    '''
    x2: center of the L1 ball (bs x input_dim)
    y2: current perturbation (x2 + y2 is the point to be projected)
    eps1: radius of the L1 ball

    output: delta s.th. ||y2 + delta||_1 <= eps1
    and 0 <= x2 + y2 + delta <= 1
    '''

    x = x2.clone().float().view(x2.shape[0], -1)
    y = y2.clone().float().view(y2.shape[0], -1)
    sigma = y.clone().sign()
    u = torch.min(1 - x - y, x + y)
    #u = torch.min(u, epsinf - torch.clone(y).abs())
    u = torch.min(torch.zeros_like(y), u)
    l = -torch.clone(y).abs()
    d = u.clone()
    
    bs, indbs = torch.sort(-torch.cat((u, l), 1), dim=1)
    bs2 = torch.cat((bs[:, 1:], torch.zeros(bs.shape[0], 1).to(bs.device)), 1)
    
    inu = 2*(indbs < u.shape[1]).float() - 1
    size1 = inu.cumsum(dim=1)
    
    s1 = -u.sum(dim=1)
    
    c = eps1 - y.clone().abs().sum(dim=1)
    c5 = s1 + c < 0
    c2 = c5.nonzero().squeeze(1)
    
    s = s1.unsqueeze(-1) + torch.cumsum((bs2 - bs) * size1, dim=1)
    
    if c2.nelement != 0:
    
      lb = torch.zeros_like(c2).float()
      ub = torch.ones_like(lb) *(bs.shape[1] - 1)
      
      #print(c2.shape, lb.shape)
      
      nitermax = torch.ceil(torch.log2(torch.tensor(bs.shape[1]).float()))
      counter2 = torch.zeros_like(lb).long()
      counter = 0
          
      while counter < nitermax:
        counter4 = torch.floor((lb + ub) / 2.)
        counter2 = counter4.type(torch.LongTensor)
        
        c8 = s[c2, counter2] + c[c2] < 0
        ind3 = c8.nonzero().squeeze(1)
        ind32 = (~c8).nonzero().squeeze(1)
        #print(ind3.shape)
        if ind3.nelement != 0:
            lb[ind3] = counter4[ind3]
        if ind32.nelement != 0:
            ub[ind32] = counter4[ind32]
        
        #print(lb, ub)
        counter += 1
        
      lb2 = lb.long()
      alpha = (-s[c2, lb2] -c[c2]) / size1[c2, lb2 + 1] + bs2[c2, lb2]
      d[c2] = -torch.min(torch.max(-u[c2], alpha.unsqueeze(-1)), -l[c2])
    
    return (sigma * d).view(x2.shape)

def L1_norm(x, keepdim=False):
    z = x.abs().view(x.shape[0], -1).sum(-1)
    if keepdim:
        z = z.view(-1, *[1]*(len(x.shape) - 1))
    return z

def L2_norm(x, keepdim=False):
    z = (x ** 2).view(x.shape[0], -1).sum(-1).sqrt()
    if keepdim:
        z = z.view(-1, *[1]*(len(x.shape) - 1))
    return z

def L0_norm(x):
    return (x != 0.).view(x.shape[0], -1).sum(-1)
def grad_normalize(source_latent,X_adv,grad,i,eps=20,step_size=10,clamp_min=-200,clamp_max=200,iters=100):
    adv_image=X_adv+step_size*grad.sign()
    eta=torch.clamp(adv_image-source_latent,-eps,eps)
    X_adv=torch.clamp(source_latent+eta,clamp_min,clamp_max).detach()
    return X_adv,(X_adv-source_latent).detach()
class AttnController:
        def __init__(self) -> None:
            self.attn_probs = []
            self.logs = []
            self.state = False
        def __call__(self, attn_prob, m_name):
            if self.state ==True:
                bs, _ = self.concept_positions.shape
                head_num = attn_prob.shape[0] // bs
                target_attns = attn_prob.masked_select(self.concept_positions[:,None,:].repeat(head_num, 1, 1)).reshape(-1, self.concept_positions[0].sum())
                self.attn_probs.append(target_attns)
                self.logs.append(m_name)
            else:
                pass
        def set_concept_positions(self, concept_positions):
            self.concept_positions = concept_positions
        def loss(self):
            return torch.cat(self.attn_probs).norm()
        def zero_attn_probs(self):
            self.attn_probs = []
            self.logs = []
            # self.concept_positions = None

class MyCrossAttnProcessor:
    def __init__(self, attn_controller: "AttnController", module_name) -> None:
        self.attn_controller = attn_controller
        self.module_name = module_name
    def __call__(self, attn: "CrossAttention", hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size=batch_size)

        query = attn.to_q(hidden_states)
        query = attn.head_to_batch_dim(query)

        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
    
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        self.attn_controller(attention_probs, self.module_name)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states
    
from typing import Any, Union
import numpy as np
from types import MethodType

def custom_set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None, inversion_flag: bool = False):
    """
    Sets the discrete timesteps used for the diffusion chain. Supporting function to be run before inference.

    Args:
        num_inference_steps (int):
            the number of diffusion steps used when generating samples with a pre-trained model.
    """

    if num_inference_steps > self.config.num_train_timesteps:
        raise ValueError(
            f"num_inference_steps: {num_inference_steps} cannot be larger than self.config.train_timesteps:"
            f" {self.config.num_train_timesteps} as the unet model trained with this scheduler can only handle"
            f" maximal {self.config.num_train_timesteps} timesteps."
        )

    self.num_inference_steps = num_inference_steps
    step_ratio = self.config.num_train_timesteps // self.num_inference_steps
    # creates integer timesteps
    timesteps = np.linspace(0, 1, num_inference_steps) * (self.config.num_train_timesteps-2) # T=999
    timesteps = timesteps + 1e-6
    timesteps = timesteps.round().astype(np.int64)
    # reverse timesteps except for inverse diffusion
    if not inversion_flag:
        timesteps = np.flip(timesteps).copy()

    self.timesteps = torch.from_numpy(timesteps).to(device)
    self.timesteps += self.config.steps_offset
class METHOD2:
    def __init__(self,model,**default_config) -> None:
        self.model = model
        self.__dict__.update(default_config)
        # num_inference_steps, lr, iters, forward_uncond_prompt, forward_cond_prompt
        self.model_prepare()
        
    def decode(self,latent):
        return self.model.decode_image(latent)
    def encode(self,image,*args,**kwds):
        return self.model.get_image_latents(image,*args,**kwds)        
    
    def custom_preprocess(self, image):
        # Do whatever you want to the image here
        return image
    
    def custom_postprocess(self, image):
        # Do whatever you want to the image here
        return image
    
    def model_prepare(self):
        for name, param in self.model.unet.named_parameters():
            param.requires_grad = False        
        attn_controller = AttnController()
        module_count = 0
        for n, m in self.model.unet.named_modules():
            if n.endswith('attn2'):
                m.set_processor(MyCrossAttnProcessor(attn_controller, n))
                module_count += 1
        concept_positions = [1] * 77
        concept_positions = torch.tensor(concept_positions,dtype=torch.bool)[None].to(self.model.device)
        attn_controller.set_concept_positions(concept_positions)
        self.attn_controller =  attn_controller
        self.model.scheduler.set_timesteps=MethodType(custom_set_timesteps,self.model.scheduler)
        self.model.scheduler.set_timesteps(self.__dict__["num_inference_steps"])
        self.forward_text_embeddings =  torch.cat([self.model.get_text_embedding(self.__dict__["forward_uncond_prompt"]),self.model.get_text_embedding(self.__dict__["forward_cond_prompt"]),self.model.get_text_embedding(self.__dict__["forward_neg_prompt"])])
        self.backward_text_embeddings =  torch.cat([self.model.get_text_embedding(self.__dict__["backward_uncond_prompt"]),self.model.get_text_embedding(self.__dict__["backward_cond_prompt"]),self.model.get_text_embedding(self.__dict__["backward_neg_prompt"])])
        self.uncond_prompt = self.model.get_text_embedding('')
    
    def compute_grad(self, source_latents,target_latents, **attack_args):
        self.loss_type=self.__dict__['loss_type']
        self.model.scheduler.set_timesteps(self.__dict__["num_inference_steps"])
        self.attn_controller.state = False
        self.attn_controller.zero_attn_probs()
        device = source_latents.device
        img_shape = source_latents.shape
        self.norm=self.__dict__['norm'] 
        self.eps=self.__dict__['eps'] 
        latent_shape = (img_shape[0], 4,img_shape[-2]//8, img_shape[-1]//8) 
        if self.norm == 'L1':
            t = torch.randn_like(source_latents).to(device).detach()
            delta = L1_projection(source_latents, t, self.eps)
            x_adv = source_latents + t + delta
        else:
            x_adv = source_latents.clone()            

        x_adv = x_adv.clamp(-1., 1.).to(dtype=source_latents.dtype)
        timestep_tensor = self.model.scheduler.timesteps.to(device)
        self.clean_X_T,self.clean_xs=self.get_clean_X_T(target_latents,timestep_tensor,img_shape,device)
        x_adv.requires_grad_()
        grad = torch.zeros_like(source_latents)
        vjp_encode_fn = lambda x: (self.model.vae.encode(x.contiguous().reshape(img_shape)).latent_dist.mode()*0.18215).view(-1)
        vjp_decode_fn = lambda x: (self.model.vae.decode(((1 / 0.18215)*x).contiguous().reshape(latent_shape)).sample).view(-1)
        with tqdm(range(self.__dict__['iters']), position=0, leave=True,desc='outer') as pbar:
            for i_inter in pbar:
                grad,loss=self.get_grad(x_adv,vjp_encode_fn,timestep_tensor,img_shape,device,vjp_decode_fn)
                x_adv,d_x_norm=grad_normalize(source_latents,x_adv,grad,i_inter,eps=self.__dict__["eps"],step_size=self.__dict__["lr"],clamp_min=self.__dict__["clamp_min"],clamp_max=self.__dict__["clamp_max"])
                pbar.set_postfix({'loss':loss.item()})
        return x_adv
    def get_X_T(self, xx,timestep_tensor,text_embedding_un,text_embedding_cond,text_embedding_neg,cfg):
        mid_in=[]
        for tt in reversed(timestep_tensor):
            mid_in.append(xx.detach().clone())
            xx = self.custom_diffusion(tt,xx,text_embedding_un,text_embedding_cond,text_embedding_neg,cfg,inversion=True)
        return mid_in,xx
        
    def custom_diffusion(self,t,z,text_embedding_un,text_embedding_cond,text_embedding_neg, cfg,inversion=False):
        """
        t: timestep
        z: latent
        
        cfg_noise =  noise_pred_0+cfg*(noise_pred_1-noise_pred_2)
        """

        if cfg >1 or cfg <0:
            if self.__dict__['using_neg']:
                model_input=torch.cat([z] * 3)
                tt=t.repeat(3).cpu()
                model_input_text = torch.stack([text_embedding_un,text_embedding_cond,text_embedding_neg])
                model_input = self.model.scheduler.scale_model_input(model_input, tt)
                
                model_pred=self.model.unet(model_input,tt.to(model_input.device),encoder_hidden_states=model_input_text).sample
                pred_noise_0,pred_noise_1,pred_noise_2 = model_pred.chunk(3)
                cfg_noise =  pred_noise_0+cfg*(pred_noise_1-pred_noise_2)                
            else:
                model_input=torch.cat([z] * 2)
                tt=t.repeat(2).cpu()
                model_input_text = torch.stack([text_embedding_un,text_embedding_cond])
                model_input = self.model.scheduler.scale_model_input(model_input, tt)
                
                model_pred=self.model.unet(model_input,tt.to(model_input.device),encoder_hidden_states=model_input_text).sample
                pred_noise_0,pred_noise_1 = model_pred.chunk(2)
                cfg_noise =  pred_noise_0+cfg*(pred_noise_1)
            prev_timestep = (tt[0]-self.model.scheduler.config.num_train_timesteps // self.model.scheduler.num_inference_steps)
            alpha_prod_t = self.model.scheduler.alphas_cumprod[tt[0]]
        elif cfg == 1:
            tt=t
            model_input_text = torch.stack([text_embedding_un])
            model_input = z
            model_input = self.model.scheduler.scale_model_input(model_input, tt)
            model_pred=self.model.unet(model_input,tt.to(z.device),encoder_hidden_states=model_input_text).sample
            pred_noise_0 = model_pred
            cfg_noise =  pred_noise_0
            prev_timestep = (tt-self.model.scheduler.config.num_train_timesteps // self.model.scheduler.num_inference_steps)
            alpha_prod_t = self.model.scheduler.alphas_cumprod[tt]
        else:
            raise ValueError('check cfg value')

        alpha_prod_t_prev = (
            self.model.scheduler.alphas_cumprod[prev_timestep]
            if prev_timestep >= 0
            else self.model.scheduler.final_alpha_cumprod
        )
        if inversion:
            alpha_prod_t, alpha_prod_t_prev = alpha_prod_t_prev, alpha_prod_t
        res_latent=backward_ddim(z, alpha_prod_t, alpha_prod_t_prev,eps_xt=cfg_noise)
        return res_latent
    
    def normalize(self, x): 
        if self.norm == 'Linf':
            t = x.abs().view(x.shape[0], -1).max(1)[0]

        elif self.norm == 'L2':
            t = (x ** 2).view(x.shape[0], -1).sum(-1).sqrt()

        elif self.norm == 'L1':
            try:
                t = x.abs().view(x.shape[0], -1).sum(dim=-1)
            except:
                t = x.abs().reshape([x.shape[0], -1]).sum(dim=-1)

        return x / (t.view(-1, *([1] * self.ndims)) + 1e-12)    
    def get_clean_X_T(self,source_latent,timestep_tensor,img_shape,device):
        x_adv=source_latent.clone()
        with torch.no_grad():
            alphas=torch.sqrt(self.model.scheduler.alphas_cumprod).to(device, dtype=x_adv.dtype)
            xx = self.model.vae.encode(x_adv).latent_dist.mode()*0.18215
            mid_x=[]
            for tt in reversed(timestep_tensor):
                mid_x.append(xx.detach().clone())
                xx = self.custom_diffusion(tt,xx,self.forward_text_embeddings[0],self.forward_text_embeddings[1],self.forward_text_embeddings[2],self.__dict__["forward_cfg"],inversion=True)
            mid_x.insert(0,x_adv)
            x_T=xx
        return x_T,mid_x
    
    def get_grad(self,x_adv,vjp_encode_fn,timestep_tensor,img_shape,device,vjp_decode_fn,latent_shape=(-1,4,64,64)):
        with torch.no_grad():
            xx = self.model.vae.encode(x_adv).latent_dist.mode()*0.18215
            mid_x=[]
            tt_bin=[]
            for iii,tt in enumerate(reversed(timestep_tensor)):
                if tt < self.__dict__['forward_point']:
                    mid_x.append(xx.detach().clone())
                    tt_bin.append(tt)
                    xx = self.custom_diffusion(tt,xx,self.forward_text_embeddings[0],self.forward_text_embeddings[1],self.forward_text_embeddings[2],self.__dict__["forward_cfg"],inversion=True)
                else:
                    break
            backward_cnt=0
            for tt2 in timestep_tensor:

                if tt2>self.__dict__['backward_point'] and tt>=tt2:
                    mid_x.append(xx.detach().clone())
                    tt_bin.append(tt2)
                    xx = self.custom_diffusion(tt2,xx,self.backward_text_embeddings[0],self.backward_text_embeddings[1],self.backward_text_embeddings[2],self.__dict__["backward_cfg"],inversion=False)
                    backward_cnt+=1
            x_T=xx
            mid_x.insert(0,x_adv)
        inputs=torch.zeros_like(x_T).to(device)
        inputs.requires_grad_(True)
        inputs.data=x_T.clone()
        if self.loss_type=='20': 
            assert self.__dict__['backward_point']==0
            loss=(self.model.decode_image(inputs).float() - x_adv.detach().float()).norm(p=2)
        print_loss=loss
        lam = torch.autograd.grad(loss, inputs, create_graph=False,retain_graph=True)[0]
        lam = lam.detach().clone()
        for idx,mid in enumerate(reversed(mid_x)):
            inputs=torch.zeros_like(x_T).to(device)
            inputs.requires_grad_(True)
            inputs.data=mid.clone()
            if idx == len(mid_x)-1: 
                vae_out,vjp=torch.autograd.functional.vjp(vjp_encode_fn,inputs.view(-1), v=lam.detach().clone().reshape(-1))
                grad=vjp.detach().clone().reshape(img_shape)
            else:
                it=tt_bin[-(idx+1)]
                if backward_cnt != 0:
                    outputs = self.custom_diffusion(it,inputs,self.backward_text_embeddings[0],self.backward_text_embeddings[1],self.backward_text_embeddings[2],self.__dict__["backward_cfg"],inversion=False)
                    backward_cnt-=1
                else:
                    outputs = self.custom_diffusion(it,inputs,self.forward_text_embeddings[0],self.forward_text_embeddings[1],self.forward_text_embeddings[2],self.__dict__["forward_cfg"],inversion=True)
                loss=torch.sum(outputs*lam)
                loss = loss 
                lam = torch.autograd.grad(loss, inputs, create_graph=True)[0].detach().clone()
        return grad,print_loss

    def attack(self, *args: Any, **kwds: Any) -> Any:
        source_latents = kwds['source_latents']
        kwds['source_latents']=self.encode(source_latents)
        target_latents = kwds['target_latents']
        kwds['target_latents']=self.encode(target_latents)
        attack_hist=self.compute_grad(*args, **kwds)
        return attack_hist
    def __call__(self, *args: Any, **kwds: Any) -> Any:    
        attack_hist=self.compute_grad(*args, **kwds)
        return attack_hist.to( kwds['source_latents'].device, kwds['source_latents'].dtype)
