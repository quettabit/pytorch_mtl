from io import open
import glob
import unicodedata
import string
from nltk import tokenize
import re
import numpy as np
import random
import pickle
import datetime
import texttable as tt

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

use_cuda = False

def printMsg(msg):
    current_time = datetime.datetime.now()
    msg = "{:%D:%H:%M:%S} ---- ".format(current_time) + msg + "\n"
    with open('test_output.txt', 'a') as f:
        f.write(msg)

SOS_token = 0
EOS_token = 1
class MetaData:
    def __init__(self):
        self.word2index = {"SOS": 0, "EOS": 1}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS
        self.max_len = -1

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def indexesFromSentence(meta_data, data):
    return [meta_data.word2index[word] for word in data.split(' ')]


def variableFromSentence(meta_data, data):
    indexes = indexesFromSentence(meta_data, data)
    indexes.append(EOS_token)
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    if use_cuda:
        return result.cuda()
    else:
        return result


def variablesFromData(data, meta_data):
    input_variable = variableFromSentence(meta_data, data[0])
    target_variable = variableFromSentence(meta_data, data[1])
    return (input_variable, target_variable)


MAX_LENGTH = 52

class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, word_embeddings, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size


        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.embedding.weight.data.copy_(torch.from_numpy(word_embeddings))
        self.linear = nn.Linear(self.embedding_size, self.hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = self.linear(embedded)
        for i in range(self.n_layers):
            output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result



class AttnDecoderRNN(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, word_embeddings, n_layers=1, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.embedding.weight.data.copy_(torch.from_numpy(word_embeddings))
        self.linear = nn.Linear(self.embedding_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)
        embedded = self.linear(embedded)


        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)))
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        for i in range(self.n_layers):
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]))
        return output, hidden, attn_weights

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result

def test(encoder, decoder, test_set, meta_data,
max_length=MAX_LENGTH, printEvery=25):

    test_pairs = [variablesFromData(sample, meta_data) for sample in test_set]
    criterion = nn.NLLLoss()
    avg_loss = 0
    nw_output = []

    for i, pair in enumerate(test_pairs):

        if (i+1) % printEvery == 0:
            printMsg("%s samples tested .." % (i))

        input_variable = pair[0]
        target_variable = pair[1]
        input_length = input_variable.size()[0]
        target_length = target_variable.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
        encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_variable[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]

        decoder_input = Variable(torch.LongTensor([[SOS_token]]))  # SOS
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

        decoder_hidden = encoder_hidden


        loss = 0
        decoded_words = []

        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]

            loss += criterion(decoder_output, target_variable[di])

            decoded_words.append(meta_data.index2word[ni])

            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input


        nw_output.append((test_set[i][0], test_set[i][1], ' '.join(decoded_words)))


        avg_loss += loss.data[0] / target_length

    test_loss = avg_loss / len(test_pairs)
    return test_loss, nw_output

def rogue1F1(outputs):
    avg_r1_f1 = 0
    r1_f1_scores = []
    for i, output in enumerate(outputs):
        ref_tokens = output[1].split(' ')
        sys_tokens = output[2].split(' ')
        tt_dict = {}
        for token in ref_tokens:
            if token in tt_dict:
                tt_dict[token] = tt_dict[token] + 1
            else:
                tt_dict[token] = 0
        num_overlaps = 0
        for token in sys_tokens:
            if token in tt_dict and tt_dict[token] > 0:
                tt_dict[token] = tt_dict[token] - 1
                num_overlaps += 1
        r1_recall = num_overlaps / len(ref_tokens)
        r1_precision = num_overlaps / len(sys_tokens)
        try:
            r1_f1 = 2*(r1_precision * r1_recall) / (r1_precision + r1_recall)
        except ZeroDivisionError:
            r1_f1 = 0
        r1_f1_scores.append((r1_f1, i))
        avg_r1_f1 += r1_f1
    return (avg_r1_f1 / len(outputs)), r1_f1_scores


def printSample(outputs):
    tab = tt.Texttable()
    headings = ['Text','Reference Summary','System Summary']
    tab.header(headings)
    for output in outputs:
        tab.add_row(list(output))
    s = tab.draw()
    printMsg(s)



if __name__ == '__main__':

    encoder = torch.load('checkpoint_models/summ_encoder_12_21_17_05_06_37_2')
    decoder = torch.load('checkpoint_models/decoder_12_21_17_05_06_37_2')
    meta_data = pickle.load(open('pickles/meta_data.pkl', 'rb'))
    test_data = pickle.load(open('pickles/test_summ_data.pkl', 'rb'))
    printMsg('models, metadata, and data loaded ..')
    printMsg('testing started ..')
    loss, nw_outputs = test(encoder.cpu().eval(), decoder.cpu().eval(), test_data, meta_data)
    printMsg('testing done ..')
    printMsg("The negative log likelihood loss for the test data is %s .." % (loss))
    avg_r1_f1, r1_f1_scores = rogue1F1(nw_outputs)
    printMsg("The average ROGUE-1 F1 measure is %s .." % (avg_r1_f1))
    sample_outputs = random.sample(nw_outputs, 10)
    printMsg("Here are some of the sample results .. ")
    printSample(sample_outputs)
