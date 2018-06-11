import datetime
import glob
import re
import random
import string
import unicodedata
from io import open

import numpy as np
import pickle
import torch
import torch.nn as nn

from nltk import tokenize
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F

USE_CUDA = False
MAX_LENGTH = 52
SPLIT_RATIOS = {'train': 80, 'validation': 10, 'test': 10}
BATCH_SIZE = 32
SOS_TOKEN = 0
EOS_TOKEN = 1

class MetaData:
    def __init__(self):
        self.word_to_index = {"SOS": 0, "EOS": 1}
        self.word_to_count = {}
        self.index_to_word = {0: "SOS", 1: "EOS"}
        self.num_words = 2  # Count SOS and EOS
        self.max_len = -1

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word_to_index:
            self.word_to_index[word] = self.num_words
            self.word_to_count[word] = 1
            self.index_to_word[self.num_words] = word
            self.num_words += 1
        else:
            self.word_to_count[word] += 1


class EncoderRNN(nn.Module):
    '''
        RNN GRU Encoder
    '''
    def __init__(self, vocab_size, embedding_size, 
                    hidden_size, word_embeddings):
        super(EncoderRNN, self).__init__()
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
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def init_hidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if USE_CUDA:
            return result.cuda()
        else:
            return result


class AttnDecoderRNN(nn.Module):
    '''
        RNN GRU Decoder with Attention  
    '''
    def __init__(self, vocab_size, embedding_size, 
                    hidden_size, word_embeddings, dropout_p=0.1, 
                    max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
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

        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], 
                                                        hidden[0]), 1)))
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                    encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)
        output = F.relu(output)

        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]))

        return output, hidden, attn_weights

    def init_hidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if USE_CUDA:
            return result.cuda()
        else:
            return result

def print_msg(msg):
    current_time = datetime.datetime.now()
    msg = "{:%D:%H:%M:%S} ---- ".format(current_time) + msg + "\n"
    with open('output.txt', 'a') as f:
        f.write(msg)

'''
load summarization and entailment datasets from pkl file
'''
summ_data = pickle.load(open('pickles/summ_data.pkl','rb'))
ent_data = pickle.load(open('pickles/ent_data.pkl','rb'))
print_msg('datasets loaded ..')

random.shuffle(summ_data)
random.shuffle(ent_data)
print_msg('datasets shuffled ..')

'''
load GloVe word embeddings
'''
word_embeddings = {}
with open('data/glove.6B.300d.txt', 'r') as f:
    for line in f:
        splits = line.split(' ')
        word = splits[0]
        embeds = splits[1:len(splits)]
        embeds = [float(embed) for embed in embeds]
        word_embeddings[word] = embeds
print_msg('embeddings loaded ..')

'''
create unified vocabulary out of summarization dataset + entailment dataset
'''
word_embedding_keys = set(list(word_embeddings.keys()))
meta_data = MetaData()

for pair in summ_data:
    meta_data.add_sentence(pair[0])
    meta_data.add_sentence(pair[1])

for pair in ent_data:
    meta_data.add_sentence(pair[0])
    meta_data.add_sentence(pair[1])

print_msg('meta data created ..')

pickle.dump(meta_data, open('pickles/meta_data.pkl', 'wb'))

print_msg('meta_data pickled ..')

vocab_size = meta_data.num_words
embedding_size = 300

'''
merge embeddings - glove embedding if present; else a normal distribution
'''
np_embeddings = np.ndarray(shape=(vocab_size, embedding_size))
for index in range(vocab_size):
    word = meta_data.index_to_word[index]
    if word in word_embedding_keys:
        np_embeddings[index] = word_embeddings[word]
    else:
        np_embeddings[index] = np.random.normal(0, 1, embedding_size)

print_msg('numpy embedding matrix created ..')


'''
helper functions to create pytorch autograd.Variables out of indexes 
in vocab mapped from/to input/output strings
'''
def indexes_from_sentence(meta_data, data):
    return [meta_data.word_to_index[word] for word in data.split(' ')]


def variable_from_sentence(meta_data, data):
    indexes = indexes_from_sentence(meta_data, data)
    indexes.append(EOS_TOKEN)
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    if USE_CUDA:
        return result.cuda()
    else:
        return result


def variables_from_data(data, meta_data):
    input_variable = variable_from_sentence(meta_data, data[0])
    target_variable = variable_from_sentence(meta_data, data[1])
    return (input_variable, target_variable)


train_summ_data_len = int(len(summ_data)*SPLIT_RATIOS['train']/100)
valid_summ_data_len = int(len(summ_data)*SPLIT_RATIOS['validation']/100)


train_summ_data = summ_data[:train_summ_data_len]
valid_summ_data = summ_data[train_summ_data_len:
                                (train_summ_data_len + valid_summ_data_len)]
test_summ_data = summ_data[(train_summ_data_len + valid_summ_data_len):]

print_msg('train/valid/test datasets created ..')

pickle.dump(train_summ_data, open('pickles/train_summ_data.pkl', 'wb'))
pickle.dump(valid_summ_data, open('pickles/valid_summ_data.pkl', 'wb'))
pickle.dump(test_summ_data, open('pickles/test_summ_data.pkl', 'wb'))

print_msg('train/valid/test datasets pickled ..')

train_summ_batches = [train_summ_data[x:x+BATCH_SIZE] 
                        for x in range(0, len(train_summ_data), BATCH_SIZE)]


train_ent_batches = [ent_data[x:x+BATCH_SIZE] 
                        for x in range(0, len(ent_data), BATCH_SIZE)]

print_msg('batch datasets created ..')


def checkpoint(summ_encoder, ent_encoder, decoder, 
                valid_loss):
    '''
        saves the encoder and decoder objects
    '''
    current_time = datetime.datetime.now()
    timestamp = "{:%D_%H_%M_%S}".format(current_time).replace('/','_')
    loss = str(valid_loss).split('.')[0]
    torch.save(summ_encoder,
                "checkpoint_models/summ_encoder_%s_%s" % (timestamp, loss))
    print_msg('summ_encoder model saved ..')
    torch.save(ent_encoder,
                "checkpoint_models/ent_encoder_%s_%s" % (timestamp, loss))
    print_msg('ent_encoder model saved ..')
    torch.save(decoder,
                "checkpoint_models/decoder_%s_%s" % (timestamp, loss))
    print_msg('decoder model saved ..')


def evaluate(encoder, decoder, validation_set, 
                meta_data, max_length=MAX_LENGTH):
    '''
        evaluates the performance of the model on the validation set
    '''

    eval_pairs = [variables_from_data(sample, meta_data) 
                    for sample in validation_set]
    criterion = nn.NLLLoss()
    avg_loss = 0

    for _, pair in enumerate(eval_pairs):

        input_variable = pair[0]
        target_variable = pair[1]
        input_length = input_variable.size()[0]
        target_length = target_variable.size()[0]
        encoder_hidden = encoder.init_hidden()

        encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
        encoder_outputs = encoder_outputs.cuda() if USE_CUDA \
                                                    else encoder_outputs

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_variable[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]

        decoder_input = Variable(torch.LongTensor([[SOS_TOKEN]])) 
        decoder_input = decoder_input.cuda() if USE_CUDA else decoder_input

        decoder_hidden = encoder_hidden


        loss = 0

        for di in range(target_length):
            decoder_output,\
                decoder_hidden,\
                decoder_attention = decoder(decoder_input, decoder_hidden, 
                                                encoder_outputs)
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]

            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input.cuda() if USE_CUDA else decoder_input

            loss += criterion(decoder_output, target_variable[di])
            if ni == EOS_TOKEN:
                break

        avg_loss += loss.data[0] / target_length


    return (avg_loss / len(eval_pairs))


def train_sample(input_variable, target_variable, encoder, 
                    decoder, encoder_optimizer, decoder_optimizer,
                    criterion, teacher_forcing_ratio, max_length=MAX_LENGTH):
    '''
        trains a single data sample
    '''

    encoder_hidden = encoder.init_hidden()

    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if USE_CUDA else encoder_outputs

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei], 
                                                    encoder_hidden)
        encoder_outputs[ei] = encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_TOKEN]]))
    decoder_input = decoder_input.cuda() if USE_CUDA else decoder_input

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio \
                                else False

    if use_teacher_forcing:

        for di in range(target_length):
            decoder_output,\
                decoder_hidden,\
                decoder_attention = decoder(decoder_input, decoder_hidden, 
                                                encoder_outputs)
            loss += criterion(decoder_output, target_variable[di])
            decoder_input = target_variable[di]

    else:

        for di in range(target_length):
            decoder_output,\
                decoder_hidden,\
                decoder_attention = decoder(decoder_input, decoder_hidden, 
                                                encoder_outputs)
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]

            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input.cuda() if USE_CUDA else decoder_input

            loss += criterion(decoder_output, target_variable[di])
            if ni == EOS_TOKEN:
                break

    loss.backward()



    return loss.data[0] / target_length


def train_batch(batch, meta_data, encoder, 
                    decoder, encoder_optimizer, 
                    decoder_optimizer, criterion):
    '''
        trains a batch of data samples
    '''
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    teacher_forcing_ratio = 0.5
    loss = 0
    training_pairs = [variables_from_data(sample, meta_data) 
                        for sample in batch]

    for pair in training_pairs:

        loss += train_sample(pair[0], pair[1],
                             encoder, decoder,
                             encoder_optimizer,
                             decoder_optimizer,
                             criterion,
                             teacher_forcing_ratio)


    encoder_optimizer.step()
    decoder_optimizer.step()

    return (loss / len(batch))


def train(train_summ_batches, valid_summ_data, train_ent_batches, 
            np_embeddings, meta_data, num_epochs, 
            pt_reload=False, switch=1, print_every=1, 
            validate_every=1, learning_rate=0.005):

    ent_batches_len = 10
    hidden_size = 512

    summ_encoder = None
    ent_encoder = None
    decoder = None

    if pt_reload:
        summ_encoder = torch.load('reloads/summ_encoder.pt')
        ent_encoder = torch.load('reloads/ent_encoder.pt')
        decoder = torch.load('reloads/decoder.pt')
        print_msg('reloading of saved models done ..')
    else:
        summ_encoder = EncoderRNN(vocab_size, embedding_size, 
                                    hidden_size, np_embeddings)
        ent_encoder = EncoderRNN(vocab_size, embedding_size, 
                                    hidden_size, np_embeddings)
        decoder = AttnDecoderRNN(vocab_size, embedding_size, 
                                    hidden_size, np_embeddings)


    if USE_CUDA:
        summ_encoder = summ_encoder.cuda()
        ent_encoder = ent_encoder.cuda()
        decoder = decoder.cuda()


    summ_encoder_optimizer = optim.Adam(summ_encoder.parameters(), 
                                            lr=learning_rate)
    ent_encoder_optimizer = optim.Adam(ent_encoder.parameters(), 
                                            lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)

    criterion = nn.NLLLoss()

    avg_train_loss_val = 0.0
    avg_train_loss_num = 0

    for epoch in range(num_epochs):
        for sbi, summ_batch in enumerate(train_summ_batches):

            try:

                loss = train_batch(summ_batch, meta_data,
                                    summ_encoder.train(),
                                    decoder.train(),
                                    summ_encoder_optimizer,
                                    decoder_optimizer,
                                    criterion)


                avg_train_loss_val += loss
                avg_train_loss_num += 1

                if (sbi+1) % print_every == 0:
                    print_msg("epoch %s, batch %s, avg. loss so far is %s .." % 
                                (epoch, sbi, 
                                    avg_train_loss_val/avg_train_loss_num ))


                if (sbi+1) % switch == 0:

                    sample_ent_batches = random.sample(train_ent_batches, 
                                                        ent_batches_len)

                    for ent_batch in sample_ent_batches:

                        train_batch(ent_batch, meta_data,
                                    ent_encoder.train(),
                                    decoder.train(),
                                    ent_encoder_optimizer,
                                    decoder_optimizer,
                                    criterion)

                    print_msg('batch mixing done ..')

                if (sbi+1) % validate_every == 0:

                    avg_valid_loss = evaluate(summ_encoder.eval(), 
                                                decoder.eval(), valid_summ_data, 
                                                meta_data)

                    print_msg("avg. validation loss is %s .." % 
                                (avg_valid_loss))

                    checkpoint(summ_encoder, ent_encoder, decoder, 
                                    avg_valid_loss)

            except Exception as e:
                print_msg("Exception at epoch %s, batch %s" % (epoch, sbi))
                print_msg(e)

        print_msg("epoch %s done, avg. train loss is %s .." % 
                    (epoch,  (avg_train_loss_val/avg_train_loss_num)))

    avg_valid_loss = evaluate(summ_encoder.eval(), decoder.eval(),
                                valid_summ_data, meta_data)

    print_msg("avg. validation loss is %s .." % (avg_valid_loss))


if __name__ == '__main__':
    print_msg('training started ..')

    train(train_summ_batches,
          valid_summ_data,
          train_ent_batches,
          np_embeddings, meta_data, 75, False)

    print_msg('training done ..')