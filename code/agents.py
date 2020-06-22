import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import os

from model_utils import RnnEncoder
from model_utils import TwoLayerMLP
from vision import Vision

from amdim.model import Model

class Agent(nn.Module):
    def __init__(self, agent_hps=None, vision_hps=None):
        super(Agent, self).__init__()
        # agent hps
        self.hidden_size = agent_hps["hidden_size"]
        self.vocab_size = agent_hps["vocab_size"]
        self.emb_size = agent_hps["emb_size"]
        self.message_length = agent_hps["message_length"]

        # shared embedding layer
        self.shared_embedding = nn.Embedding(self.vocab_size, self.emb_size)
        self.sos_embedding = nn.Parameter(torch.zeros(self.emb_size))
        
        # shared vision + linear layers
        self.input_channels = vision_hps["input_channels"]
        self.vision_ckpt = vision_hps["vision_ckpt"]
        self.vision = Vision(self.input_channels)
        self.fc = nn.Linear(576, self.hidden_size)
        
        # sender modules
        self.sender_decoder = nn.LSTMCell(self.emb_size, self.hidden_size)
        self.sender_hidden_to_output = nn.Linear(self.hidden_size, self.vocab_size)

        # receiver modules
        self.receiver_encoder = RnnEncoder(self.vocab_size, self.shared_embedding, self.hidden_size, "lstm")
        
    def forward(self, mode, **kwargs):
        if mode == "sender":
            output = self.sender_forward(tgt_img=kwargs["tgt_img"])  # message
        
        elif mode == "receiver":
            output = self.receiver_forward(imgs=kwargs["imgs"], message=kwargs["message"])  # prediction
        
        return output

    def sender_forward(self, tgt_img):
        enc_outputs = self.vision(tgt_img)
        feature_vector = enc_outputs.view(enc_outputs.size(0), -1)  # b * features
        ht = self.fc(feature_vector) # b * 1 * self.hidden_size

        ct = torch.zeros_like(ht)
        et = torch.stack([self.sos_embedding] * ht.size(0))

        message = []
        log_probs = []
        entropy = []

        for i in range(self.message_length - 1):
            ht, ct = self.sender_decoder(et, (ht, ct))

            step_logits = F.softmax(self.sender_hidden_to_output(ht), dim=1)
            distr = Categorical(probs=step_logits)

            if self.training:
                token = distr.sample()
            else:
                token = step_logits.argmax(dim=1)

            et = self.shared_embedding(token)

            message.append(token)
            log_probs.append(distr.log_prob(token))
            entropy.append(distr.entropy())

        message = torch.stack(message).permute(1, 0)
        log_probs = torch.stack(log_probs).permute(1, 0)
        entropy = torch.stack(entropy).permute(1, 0)

        zeros = torch.zeros((message.size(0), 1)).to(message.device)
        message = torch.cat([message, zeros.long()], dim=1)
        log_probs = torch.cat([log_probs, zeros], dim=1)
        entropy = torch.cat([entropy, zeros], dim=1)

        return message, log_probs, entropy
        
    def receiver_forward(self, message, imgs):
        batch_size = message.size(0)
        num_imgs = imgs.size(1)
        imgs = imgs.view(batch_size*num_imgs, self.input_channels, 64, 64)

        enc_outputs = self.vision(imgs)
        feature_vectors = enc_outputs.view(batch_size*num_imgs, -1) # b*num_imgs * features
        feature_vectors = self.fc(feature_vectors) # b*num_imgs * self.hidden_size
        feature_vectors = feature_vectors.view(batch_size, num_imgs, -1) # b * num_imgs * 1 * self.hidden_size

        emb_msg = self.receiver_encoder(message).unsqueeze(1)
        img_msg = torch.Tensor([]).to(feature_vectors.device)

        for i in range(num_imgs):

            # compute img/message similarity score
            img_msg = torch.cat((img_msg, torch.bmm(feature_vectors[:, i, :].unsqueeze(1), torch.transpose(emb_msg, 2, 1))), 1)
            probs = F.softmax(img_msg, 1).squeeze(-1)
        
        distr = Categorical(probs=probs)

        if self.training:
            choice = distr.sample()
        
        else:
            choice = probs.argmax(dim=1)
        
        log_probs = distr.log_prob(choice)
        entropy = None
        
        return choice, log_probs, entropy

    def load_vision(self):
        ckpt = torch.load(self.vision_ckpt)
        hp = ckpt["hyperparams"]
        params = ckpt["model"]
        
        model = Model(ndf=hp["ndf"], n_classes=hp["n_classes"], n_rkhs=hp["n_rkhs"],
        n_depth=hp["n_depth"], encoder_size=hp["encoder_size"]) 
        
        model.load_state_dict(params)
        self.vision = model.encoder

        print("Loaded checkpoint from {:s}.".format(self.vision_ckpt))
        