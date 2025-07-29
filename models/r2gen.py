import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from modules.encoder_decoder import EncoderDecoder


class R2GenModel(nn.Module):
    def __init__(self, args, encoder_decoder=None):
        super(R2GenModel, self).__init__()
        self.args = args
        self.prompt = nn.Parameter(torch.randn(1, 1, args.d_vf))
        self.fc = nn.Sequential(nn.LayerNorm(args.d_model),nn.Linear(args.d_model,args.d_model),nn.Linear(args.d_model,args.n_classes))
        if not encoder_decoder:
            print('use encoder_decoder: default')
            self.encoder_decoder = EncoderDecoder(args)
            
        if args.dataset_name:
            self.forward = self.forward_brca
        else:
            raise ValueError('no forward function')

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward_brca(self, images, text_features,targets=None, mode='train'):

        att_feats = images  # shape 1*N*384
        att_feats = torch.cat([self.prompt,att_feats],dim=1)
        fc_feats = torch.sum(att_feats,dim=1) #shape 1*384

        if mode == 'train':
            output = self.encoder_decoder(att_feats,text_features)
            final_features = output.mean(1)
            logits = self.fc(final_features)
            # print(logits.shape)
            # logits = self.fc(output[0].max(dim=0)[0]).unsqueeze(0)
            # Y_hat = torch.argmax(logits, dim=1)
            Y_hat = torch.argmax(logits, dim=1)
            Y_prob = F.softmax(logits, dim=1)
            # Y_prob = torch.sigmoid(logits)
            # Y_hat = (Y_prob > 0.5).float()
            return logits, Y_hat, Y_prob
        elif mode == 'sample':
            output, _ = self.encoder_decoder(att_feats, mode='sample')
        elif mode == 'encode':
            output = self.encoder_decoder(fc_feats, att_feats, mode='encode')

            logits = self.fc(output[0,0,:]).unsqueeze(0)
            Y_hat = torch.argmax(logits, dim=1)
            Y_prob = F.softmax(logits, dim=1)
            return logits,Y_hat, Y_prob
        else:
            raise ValueError
        return output

