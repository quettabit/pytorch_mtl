import random
from io import open

import pickle
import torch
import torch.nn as nn
from torch.autograd import Variable
import texttable as tt

from mtl_learning import MetaData, EncoderRNN, AttnDecoderRNN
from mtl_learning import print_msg, variables_from_data

USE_CUDA = False
MAX_LENGTH = 52
SPLIT_RATIOS = {'train': 80, 'validation': 10, 'test': 10}
BATCH_SIZE = 32
SOS_TOKEN = 0
EOS_TOKEN = 1

def test(encoder, decoder, test_set, 
            meta_data, max_length=MAX_LENGTH, printEvery=25):

    test_pairs = [variables_from_data(sample, meta_data) for sample in test_set]
    criterion = nn.NLLLoss()
    avg_loss = 0
    nw_output = []

    for i, pair in enumerate(test_pairs):

        if (i+1) % printEvery == 0:
            print_msg("%s samples tested .." % (i))

        input_variable = pair[0]
        target_variable = pair[1]
        input_length = input_variable.size()[0]
        target_length = target_variable.size()[0]
        encoder_hidden = encoder.init_hidden()

        encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
        encoder_outputs = encoder_outputs.cuda() if USE_CUDA else encoder_outputs

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_variable[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]

        decoder_input = Variable(torch.LongTensor([[SOS_TOKEN]]))  
        decoder_input = decoder_input.cuda() if USE_CUDA else decoder_input

        decoder_hidden = encoder_hidden


        loss = 0
        decoded_words = []

        for di in range(target_length):
            decoder_output,\
                decoder_hidden,\
                decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]

            loss += criterion(decoder_output, target_variable[di])
            if ni == EOS_TOKEN:
                break
            else:
                decoded_words.append(meta_data.index_to_word[ni])

            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input.cuda() if USE_CUDA else decoder_input


        nw_output.append((test_set[i][0], test_set[i][1], ' '.join(decoded_words)))


        avg_loss += loss.data[0] / target_length

    test_loss = avg_loss / len(test_pairs)
    return test_loss, nw_output

def rogue1_F1(outputs):
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


def print_sample(outputs):
    tab = tt.Texttable()
    headings = ['Text','Reference Summary','System Summary']
    tab.header(headings)
    for output in outputs:
        tab.add_row(list(output))
    s = tab.draw()
    print_msg(s)



if __name__ == '__main__':

    encoder = torch.load('checkpoint_models/summ_encoder_06_10_18_22_37_14_1')
    decoder = torch.load('checkpoint_models/decoder_06_10_18_22_37_14_1')
    meta_data = pickle.load(open('pickles/meta_data.pkl', 'rb'))
    test_data = pickle.load(open('pickles/test_summ_data.pkl', 'rb'))
    print_msg('models, metadata, and data loaded ..')
    print_msg('testing started ..')
    loss, nw_outputs = test(encoder.eval(), decoder.eval(), 
                                test_data, meta_data)
    print_msg('testing done ..')
    print_msg("The negative log likelihood loss for the test data is %s .." % 
                (loss))
    avg_r1_f1, r1_f1_scores = rogue1_F1(nw_outputs)
    print_msg("The average ROGUE-1 F1 measure is %s .." % (avg_r1_f1))
    sample_outputs = random.sample(nw_outputs, 10)
    print_msg("Here are some of the sample results .. ")
    print_sample(sample_outputs)
