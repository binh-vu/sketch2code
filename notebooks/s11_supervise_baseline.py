import torch
import torch.nn as nn
import torch.nn.functional as F

from sketch2code.methods.dqn import conv2d_size_out, pool2d_size_out
from sketch2code.methods.attention_lstm import AttentionLSTM
from s10_supervise_baseline import BLSuperV1


class AttentionEncoderV1(nn.Module):
    
    def __init__(self, img_h: int, img_w: int, encoder_dim: int, encoded_img_h: int, encoded_img_w: int):
        super().__init__()
        self.img_w = img_w
        self.img_h = img_h
        self.encoder_dim = encoder_dim
        self.encoded_img_h = encoded_img_h
        self.encoded_img_w = encoded_img_w

        self.__build_model()

    def __build_model(self):
        # network compute features of target image
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.bn1 = nn.BatchNorm2d(32, momentum=0.9)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1)
        self.bn2 = nn.BatchNorm2d(64, momentum=0.9)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3 = nn.Conv2d(64, self.encoder_dim, kernel_size=5, stride=1)
        self.bn3 = nn.BatchNorm2d(self.encoder_dim, momentum=0.9)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        if self.encoded_img_h is None:
            self.adaptive_pool = lambda x: x
        else:
            self.adaptive_pool = nn.AdaptiveAvgPool2d((self.encoded_img_h, self.encoded_img_w))

    def compute_act_layers(self, x):
        """Compute activation at different layers of an image X (C x H x W)"""
        with torch.no_grad():
            x = x.view(1, *x.shape)
            x1 = F.selu(self.bn1(self.conv1(x)))
            x1pool = self.pool1(x1)
            x2 = F.selu(self.bn2(self.conv2(x1pool)))
            x2pool = self.pool2(x2)
            x3 = F.selu(self.bn3(self.conv3(x2pool)))
            x3pool = self.pool3(x3)

            return {
                'layer1': x1[0],
                'layer1pool': x1pool[0],
                'layer2': x2[0],
                'layer2pool': x2pool[0],
                'layer3': x3[0],
                'layer3pool': x3pool[0],
            }

    def forward(self, x1):
        """
        :param x1: desired iamges (batch: N x C x H x W)
        :return:
        """
        x1 = self.pool1(F.selu(self.bn1(self.conv1(x1))))
        x1 = self.pool2(F.selu(self.bn2(self.conv2(x1))))
        x1 = self.pool3(F.selu(self.bn3(self.conv3(x1))))
        x1 = self.adaptive_pool(x1)
        # reshape to N x H x W x C
        x1 = x1.permute(0, 2, 3, 1)
        return x1

class AttentionDecoderV1(nn.Module):

    def __init__(self, encoder_dim: int, attention_dim: int, dsl_vocab, dsl_hidden_dim, dsl_embedding_dim, padding_idx: int=0):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.dsl_vocab = dsl_vocab
        self.dsl_hidden_dim = dsl_hidden_dim
        self.dsl_embedding_dim = dsl_embedding_dim
        self.padding_idx = padding_idx

        self.__build_model()

    def __build_model(self):
        self.attention_lstm = AttentionLSTM(
            encoder_dim=self.encoder_dim, 
            attention_dim=self.attention_dim, 
            embed_dim=self.dsl_embedding_dim, decoder_dim=self.dsl_hidden_dim, 
            vocab_size=len(self.dsl_vocab))
        self.lstm2token = nn.Linear(self.dsl_hidden_dim, len(self.dsl_vocab))

    def forward(self, x1, x2, x2_lens):
        """
        :param x1: output from encoder program (batch: N x E)
        :param x2: current programs (batch: N x T)
        :param x2_lens: lengths of current programs (N)
        :return:
        """
        batch_size = x2.shape[0]
        x2, alphas = self.attention_lstm(x1, x2, x2_lens)
        
        # flatten from N x T x H to (N * T) x H
        x2 = x2.contiguous().view(-1, self.dsl_hidden_dim)
        nts = F.log_softmax(self.lstm2token(x2), dim=1)
        nts = nts.view(batch_size, -1, nts.shape[-1])  # N x T x H
        
        return nts, alphas
    
class AttentionBLSuperV1(nn.Module):
    
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x1, x2, x2lens, return_alphas: bool=False):
        x1 = self.encoder(x1)
        nx2, alphas = self.decoder(x1, x2, x2lens)
        
        if return_alphas:
            return nx2, alphas
        return nx2