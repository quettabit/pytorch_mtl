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

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()


SOS_token = 0
EOS_token = 1

def printMsg(msg):
    current_time = datetime.datetime.now()
    msg = "{:%D:%H:%M:%S} ---- ".format(current_time) + msg + "\n"
    with open('output.txt', 'a') as f:
        f.write(msg)

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



# load summarization and entailment datasets from pkl file
summ_data = pickle.load(open('pickles/summ_data.pkl','rb'))
ent_data = pickle.load(open('pickles/ent_data.pkl','rb'))
printMsg('datasets loaded ..')

random.shuffle(summ_data)
random.shuffle(ent_data)
printMsg('datasets shuffled ..')

# load GloVe word embeddings
word_embeddings = {}
with open('data/glove.6B.300d.txt', 'r') as f:
    for line in f:
        splits = line.split(' ')
        word = splits[0]
        embeds = splits[1:len(splits)]
        embeds = [float(embed) for embed in embeds]
        word_embeddings[word] = embeds
printMsg('embeddings loaded ..')

# create unified vocabulary out of summarization dataset + entailment dataset
word_embedding_keys = set(list(word_embeddings.keys()))
meta_data = MetaData()

for pair in summ_data:
    meta_data.addSentence(pair[0])
    meta_data.addSentence(pair[1])

for pair in ent_data:
    meta_data.addSentence(pair[0])
    meta_data.addSentence(pair[1])

printMsg('meta data created ..')

pickle.dump(meta_data, open('pickles/meta_data.pkl', 'wb'))

printMsg('meta_data pickled ..')



vocab_size = meta_data.n_words
embedding_size = 300
#merge embeddings - glove embedding if present; else a uniform distribution

np_embeddings = np.ndarray(shape=(vocab_size, embedding_size))
for index in range(vocab_size):
    word = meta_data.index2word[index]
    if word in word_embedding_keys:
        np_embeddings[index] = word_embeddings[word]
    else:
        np_embeddings[index] = np.random.uniform(-1, 1, embedding_size)

printMsg('numpy embedding matrix created ..')





# helper functions to create pytorch autograd.Variables out of indexes in vocab mapped from input/output strings
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

# Encoder
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


#Decoder with Attention
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



SPLIT_RATIOS = {'train': 80, 'validation': 10, 'test': 10}
BATCH_SIZE = 32


train_summ_data_len = int(len(summ_data)*SPLIT_RATIOS['train']/100)
valid_summ_data_len = int(len(summ_data)*SPLIT_RATIOS['validation']/100)


train_summ_data = summ_data[:train_summ_data_len]
valid_summ_data = summ_data[train_summ_data_len: train_summ_data_len + valid_summ_data_len]
test_summ_data = summ_data[train_summ_data_len + valid_summ_data_len:]

printMsg('train/valid/test datasets created ..')

pickle.dump(train_summ_data, open('pickles/train_summ_data.pkl', 'wb'))
pickle.dump(valid_summ_data, open('pickles/valid_summ_data.pkl', 'wb'))
pickle.dump(test_summ_data, open('pickles/test_summ_data.pkl', 'wb'))

printMsg('train/valid/test datasets pickled ..')

train_summ_batches = [train_summ_data[x:x+BATCH_SIZE] for x in range(0, len(train_summ_data), BATCH_SIZE)]


train_ent_batches = [ent_data[x:x+BATCH_SIZE] for x in range(0, len(ent_data), BATCH_SIZE)]

printMsg('batch datasets created ..')

# method to save the encoders and decoder
def checkpoint(summ_encoder, ent_encoder, decoder, valid_loss):
    current_time = datetime.datetime.now()
    timestamp = "{:%D_%H_%M_%S}".format(current_time).replace('/','_')
    loss = str(valid_loss).split('.')[0]
    torch.save(summ_encoder,
                "checkpoint_models/summ_encoder_%s_%s" % (timestamp, loss))
    printMsg('summ_encoder model saved ..')
    torch.save(ent_encoder,
                "checkpoint_models/ent_encoder_%s_%s" % (timestamp, loss))
    printMsg('ent_encoder model saved ..')
    torch.save(decoder,
                "checkpoint_models/decoder_%s_%s" % (timestamp, loss))
    printMsg('decoder model saved ..')

# method to evaluate the performance of the model on the validation set
def evaluate(encoder, decoder, validation_set, meta_data,
max_length=MAX_LENGTH):

    eval_pairs = [variablesFromData(sample, meta_data) for sample in validation_set]
    criterion = nn.NLLLoss()
    avg_loss = 0

    for i, pair in enumerate(eval_pairs):

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

        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]

            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

            loss += criterion(decoder_output, target_variable[di])
            if ni == EOS_token:
                break

        avg_loss += loss.data[0] / target_length


    return (avg_loss / len(eval_pairs))

# method to train a single sample
def train_sample(input_variable, target_variable,
                 encoder, decoder, encoder_optimizer,
                 decoder_optimizer, criterion,
                 teacher_forcing_ratio,
                 max_length=MAX_LENGTH):

    encoder_hidden = encoder.initHidden()

    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_variable[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:

        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_variable[di])
            decoder_input = target_variable[di]

    else:

        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]

            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

            loss += criterion(decoder_output, target_variable[di])
            if ni == EOS_token:
                break

    loss.backward()



    return loss.data[0] / target_length

# method to train a batch of samples
def train_batch(batch, meta_data, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    teacher_forcing_ratio = 0.5
    loss = 0
    training_pairs = [variablesFromData(sample, meta_data) for sample in batch]

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

# training method
def train(train_summ_batches, valid_summ_data, train_ent_batches, np_embeddings, meta_data, n_epochs, pt_reload=False, switch=100, print_every=25, learning_rate=0.005):

    ent_batches_len = 10
    hidden_size = 512
    validate_every = 150

    summ_encoder = None
    ent_encoder = None
    decoder = None

    if pt_reload:
        summ_encoder = torch.load('reloads/summ_encoder.pt')
        ent_encoder = torch.load('reloads/ent_encoder.pt')
        decoder = torch.load('reloads/decoder.pt')
        printMsg('reloading of saved models done ..')
    else:
        summ_encoder = EncoderRNN(vocab_size, embedding_size, hidden_size, np_embeddings)
        ent_encoder = EncoderRNN(vocab_size, embedding_size, hidden_size, np_embeddings)
        decoder = AttnDecoderRNN(vocab_size, embedding_size, hidden_size, np_embeddings)


    if use_cuda:
        summ_encoder = summ_encoder.cuda()
        ent_encoder = ent_encoder.cuda()
        decoder = decoder.cuda()


    summ_encoder_optimizer = optim.Adam(summ_encoder.parameters(), lr=learning_rate)
    ent_encoder_optimizer = optim.Adam(ent_encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()


    valid_loss_history = []

    avg_train_loss_val = 0.0
    avg_train_loss_num = 0
    best_valid_loss = 6.00

    for epoch in range(n_epochs):

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
                    printMsg("epoch %s, batch %s, avg. loss so far is %s .." % (epoch, sbi, avg_train_loss_val/avg_train_loss_num ))


                if (sbi+1) % switch == 0:

                    sample_ent_batches = random.sample(train_ent_batches, ent_batches_len)

                    for ent_batch in sample_ent_batches:

                        train_batch(ent_batch, meta_data,
                                    ent_encoder.train(),
                                    decoder.train(),
                                    ent_encoder_optimizer,
                                    decoder_optimizer,
                                    criterion)

                    printMsg('batch mixing done ..')

                if (sbi+1) % validate_every == 0:

                    avg_valid_loss = evaluate(summ_encoder.eval(), decoder.eval(),
                                                valid_summ_data, meta_data)

                    printMsg("evaluation on validation set done .. avg. loss is %s .." % (avg_valid_loss))

                    valid_loss_history.append(avg_valid_loss)
                    num_valid_losses = len(valid_loss_history)

                    if num_valid_losses > 4:
                        if ( valid_loss_history[num_valid_losses-1]  <  valid_loss_history[num_valid_losses-2] ) and \
                           ( valid_loss_history[num_valid_losses-1] <
                        valid_loss_history[num_valid_losses-3] ) and \
                           ( valid_loss_history[num_valid_losses-1] <
                        best_valid_loss ):
                            best_valid_loss = avg_valid_loss
                            checkpoint(summ_encoder, ent_encoder,
                                        decoder, avg_valid_loss)

            except Exception as e:
                printMsg("Exception at epoch %s, batch %s" % (epoch, sbi))
                printMsg(e)

        printMsg("epoch %s done, avg. train loss is %s .." % (epoch,  (avg_train_loss_val/avg_train_loss_num)))

    avg_valid_loss = evaluate(summ_encoder.eval(), decoder.eval(),
                                valid_summ_data, meta_data)

    printMsg("evaluation on validation set done .. avg loss is %s .." % (avg_valid_loss))

    valid_loss_history.append(avg_valid_loss)
    num_valid_losses = len(valid_loss_history)

    if ( valid_loss_history[num_valid_losses-1]  <  valid_loss_history[num_valid_losses-2] ) and \
       ( valid_loss_history[num_valid_losses-1] <
    valid_loss_history[num_valid_losses-3] ) and \
       ( valid_loss_history[num_valid_losses-1] <
    best_valid_loss ):
        best_valid_loss = avg_valid_loss
        checkpoint(summ_encoder, ent_encoder,
                    decoder, avg_valid_loss)

    printMsg("best validation loss is %s .." % (best_valid_loss))


if __name__ == '__main__':
    printMsg('training started ..')

    train(train_summ_batches,
          valid_summ_data,
          train_ent_batches,
          np_embeddings, meta_data, 75, True)

    printMsg('training done ..')
