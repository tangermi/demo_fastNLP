# -*- coding:utf-8 -*-
import torch
import torch.nn as nn

# 构建DPCNN神经网络


class ResnetBlock(nn.Module):
    def __init__(self, channel_size):
        super(ResnetBlock, self).__init__()

        self.channel_size = channel_size
        self.maxpool = nn.Sequential(
            nn.ConstantPad1d(padding=(0, 1), value=0),
            nn.MaxPool1d(kernel_size=3, stride=2)
        )
        self.conv = nn.Sequential(
            nn.BatchNorm1d(num_features=self.channel_size),
            nn.ReLU(),
            nn.Conv1d(self.channel_size, self.channel_size, kernel_size=3, padding=1),

            nn.BatchNorm1d(num_features=self.channel_size),
            nn.ReLU(),
            nn.Conv1d(self.channel_size, self.channel_size, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x_shortcut = self.maxpool(x)
        x = self.conv(x_shortcut)
        x = x + x_shortcut
        return x


class DPCNN(nn.Module):
    def __init__(self, max_features, word_embedding_dimension, max_sentence_length, num_classes):
        super(DPCNN, self).__init__()
        self.max_features = max_features
        self.embed_size = word_embedding_dimension
        self.maxlen = max_sentence_length
        self.num_classes = num_classes
        self.channel_size = 250

        self.embedding = nn.Embedding(self.max_features, self.embed_size)
        torch.nn.init.normal_(self.embedding.weight.data, mean=0, std=0.01)
        self.embedding.weight.requires_grad = True

        # region embedding
        self.region_embedding = nn.Sequential(
            nn.Conv1d(self.embed_size, self.channel_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=self.channel_size),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.conv_block = nn.Sequential(
            nn.BatchNorm1d(num_features=self.channel_size),
            nn.ReLU(),
            nn.Conv1d(self.channel_size, self.channel_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=self.channel_size),
            nn.ReLU(),
            nn.Conv1d(self.channel_size, self.channel_size, kernel_size=3, padding=1),
        )

        self.seq_len = self.maxlen
        resnet_block_list = []
        while (self.seq_len > 2):
            resnet_block_list.append(ResnetBlock(self.channel_size))
            self.seq_len = self.seq_len // 2
        #         print('seqlen{}'.format(self.seq_len))
        self.resnet_layer = nn.Sequential(*resnet_block_list)

        self.fc = nn.Sequential(
            nn.Linear(self.channel_size * self.seq_len, self.num_classes),
            nn.BatchNorm1d(self.num_classes),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(self.num_classes, self.num_classes)
        )

    def forward(self, description_word_seq):
        x = self.embedding(description_word_seq)
        x = x.permute(0, 2, 1)
        x = self.region_embedding(x)
        x = self.conv_block(x)
        x = self.resnet_layer(x)
        x = x.permute(0, 2, 1)
        x = x.contiguous().view(x.size(0), -1)
        output = self.fc(x)
        return {'output': output}

    def predict(self, description_word_seq):
        """
        :param word_seq: torch.LongTensor, [batch_size, seq_len]
        :return predict: dict of torch.LongTensor, [batch_size, seq_len]
        """
        output = self(description_word_seq)
        _, predict = output['output'].max(dim=1)
        return {'predict': predict}