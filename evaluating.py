import os
import json
import argparse
import torch
import pandas as pd
import numpy as np
import random
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from dataset import PlainTextDataset
from modeling import LSTMVAE, BERTVAE
from nltk.translate.bleu_score import corpus_bleu


def basic(model, test_dataloader, args):
    if not os.path.exists(os.path.join('model', 'basic.txt')):
        df = pd.DataFrame(columns=['Model', 'ELBO', 'KL', 'Rec.', 'AU'])
        df.to_csv(os.path.join('model', 'basic.txt'), index=False, float_format='%.6f')
    df = pd.read_csv(os.path.join('model', 'basic.txt'))

    if os.path.basename(os.path.dirname(args.load_model_path)) in list(df['Model']):
        print('Results already exists.')
    else:
        dic = {'Model': os.path.basename(os.path.dirname(args.load_model_path))}
        total_loss = 0
        kld_loss = 0
        rec_loss = 0
        with torch.no_grad():
            for step, batch_data in enumerate(test_dataloader):
                if args.is_bert_encoder:
                    x = tokenizer(batch_data, max_length=args.max_length, padding='max_length', truncation=True,
                                  return_tensors='pt')
                    x = x.to(args.device)
                else:
                    x = tokenizer(batch_data, max_length=args.max_length, padding='max_length', truncation=True,
                                  return_token_type_ids=False, return_attention_mask=False, return_tensors='pt')
                    x = x['input_ids'].to(args.device)
                if step == 0:
                    all_mean = model.encoder(x)[1]
                else:
                    mean = model.encoder(x)[1]
                    all_mean = torch.cat([all_mean, mean], dim=0)
                loss, kld, rec = model(x)
                total_loss += loss.item()
                kld_loss += kld.item()
                rec_loss += rec.item()
            all_mean = all_mean.cpu().numpy()

        elbo, kl, rec = total_loss / (step + 1), kld_loss / (step + 1), rec_loss / (step + 1)
        print("elbo:{:.4f} kl_loss:{:.4f} rec_loss:{:.4f}".format(elbo, kl, rec))
        dic['ELBO'] = float(elbo)
        dic['KL'] = float(kl)
        dic['Rec.'] = float(rec)

        cov = np.cov(all_mean, rowvar=False)
        s = []
        n = []
        for i in range(0, cov.shape[0]):
            if cov[i][i] > 0.01:
                s.append(i + 1)
            else:
                n.append(i + 1)
        print('{} active units:{}'.format(len(s), s))
        print('{} inactive units:{}'.format(len(n), n))
        dic['AU'] = len(s)
        df = pd.concat([df, pd.DataFrame(dic, index=[0])], ignore_index=True)
        df.to_csv(os.path.join('model', 'basic.txt'), index=False, float_format='%.6f')


def get_representations(model, test_dataloader, args):
    print("get mean and sample representations for sentences")
    with torch.no_grad():
        for step, batch_data in enumerate(test_dataloader):
            if args.is_bert_encoder:
                x = tokenizer(batch_data, max_length=args.max_length, padding='max_length', truncation=True,
                              return_tensors='pt')
                x = x.to(args.device)
            else:
                x = tokenizer(batch_data, max_length=args.max_length, padding='max_length', truncation=True,
                              return_token_type_ids=False, return_attention_mask=False, return_tensors='pt')
                x = x['input_ids'].to(args.device)
            if step == 0:
                all_sample, all_mean = model.encoder(x)[0:2]
            else:
                sample, mean = model.encoder(x)[0:2]
                all_sample, all_mean = torch.cat([all_sample, sample], dim=0), torch.cat([all_mean, mean], dim=0)
        all_mean, all_sample = all_mean.cpu().numpy(), all_sample.cpu().numpy()
    mean_df, sample_df = pd.DataFrame(all_mean), pd.DataFrame(all_sample)
    mean_df.columns = ['dim' + str(i) for i in range(1, args.z_dim + 1)]
    sample.columns = ['dim' + str(i) for i in range(1, args.z_dim + 1)]
    mean_df.to_csv(os.path.join(os.path.dirname(args.load_model_path), 'means.csv'), index_label='index')
    sample_df.to_csv(os.path.join(os.path.dirname(args.load_model_path), 'sample.csv'), index_label='index')


def reconstruction(model, tokenizer, test_dataloader, args):
    if not os.path.exists(os.path.join('model', 'bleu.txt')):
        df = pd.DataFrame(columns=['Model', 'BLEU-1', 'BLEU-2', 'BLEU-4'])
        df.to_csv(os.path.join('model', 'bleu.txt'), index=False, float_format='%.6f')
    df = pd.read_csv(os.path.join('model', 'bleu.txt'))

    if os.path.basename(os.path.dirname(args.load_model_path)) + '_' + args.decoding in list(df['Model']):
        print('Results already exists.')
    else:
        dic = {'Model': os.path.basename(os.path.dirname(args.load_model_path)) + '_' + args.decoding}
        reconstruction_file = os.path.join(os.path.dirname(args.load_model_path), args.decoding + '_mean.txt')
        f = open(reconstruction_file, 'w')
        with torch.no_grad():
            for batch_data in test_dataloader:
                if args.is_bert_encoder:
                    x = tokenizer(batch_data, max_length=args.max_length, padding='max_length', truncation=True,
                                  return_tensors='pt')
                    x = x.to(args.device)
                else:
                    x = tokenizer(batch_data, max_length=args.max_length, padding='max_length', truncation=True,
                                  return_token_type_ids=False, return_attention_mask=False, return_tensors='pt')
                    x = x['input_ids'].to(args.device)
                mean = model.encoder(x)[1]
                if args.decoding == 'greedy':
                    res = model.decoder.greedy_decoding(mean, args.max_length)
                elif args.decoding == 'beam_search':
                    res = model.decoder.beam_search_decoding(mean, args.max_length)
                else:
                    raise ValueError('Invalid decoding strategy!')
                for element in res:
                    if 102 in element:
                        element = element[:element.index(102)]
                    f.write(tokenizer.decode(element) + '\n')
        f.close()
        bleu_references = []
        with open(args.test_path, 'r') as f:
            for sentence in f.readlines():
                bleu_references.append([tokenizer.tokenize(sentence.rstrip())])

        bleu_candidates = []
        with open(reconstruction_file, 'r') as f:
            for sentence in f.readlines():
                bleu_candidates.append(tokenizer.tokenize(sentence.rstrip()))

        dic['BLEU-1'] = corpus_bleu(bleu_references, bleu_candidates, weights=(1, 0, 0, 0))
        dic['BLEU-2'] = corpus_bleu(bleu_references, bleu_candidates, weights=(0.5, 0.5, 0, 0))
        dic['BLEU-4'] = corpus_bleu(bleu_references, bleu_candidates, weights=(0.25, 0.25, 0.25, 0.25))
        df = pd.concat([df, pd.DataFrame(dic, index=[0])], ignore_index=True)
        df.to_csv(os.path.join('model', 'bleu.txt'), index=False, float_format='%.6f')


def homotopy(model, tokenizer, args):
    sentences = []
    with open(args.test_path, 'r') as f:
        for sentence in f.readlines():
            if '[UNK]' not in sentence:
                sentences.append(sentence.rstrip())
    sentences = random.sample(sentences, 2)
    f = open(os.path.join(os.path.dirname(args.load_model_path), args.decoding + '_homotopy.txt'), 'w')
    f.write('start sentence: ')
    f.write(sentences[0] + '\n')
    f.write('end sentence: ')
    f.write(sentences[1] + '\n')

    with torch.no_grad():
        if args.is_bert_encoder:
            z1 = model.encoder(tokenizer([sentences[0]], return_tensors='pt').to(args.device))[1]
            z2 = model.encoder(tokenizer([sentences[1]], return_tensors='pt').to(args.device))[1]
        else:
            z1 = model.encoder(torch.tensor([tokenizer.encode(sentences[0])], device=args.device))[1]
            z2 = model.encoder(torch.tensor([tokenizer.encode(sentences[1])], device=args.device))[1]

        f.write('z1: ')
        f.write(' '.join(['%.3f' % z1.cpu().numpy().tolist()[0][i] for i in range(0, len(z1.cpu().numpy().tolist()[0]))]) + '\n')
        f.write('z2: ')
        f.write(' '.join(['%.3f' % z2.cpu().numpy().tolist()[0][i] for i in range(0, len(z2.cpu().numpy().tolist()[0]))]) + '\n')

        f.write('normal homotopy:\n')
        for i in range(0, 6):
            f.write(str(i + 1) + '. ')
            z = (1 - 0.2 * i) * z1 + 0.2 * i * z2
            if args.decoding == 'greedy':
                res = model.decoder.greedy_decoding(z, args.max_length)[0]
            elif args.decoding == 'beam_search':
                res = model.decoder.beam_search_decoding(z, args.max_length)[0]
            else:
                raise ValueError('Invalid decoding strategy!')
            if 102 in res:
                res = res[:res.index(102)]
            f.write(tokenizer.decode(res) + '\n')

        for dim in range(0, z1.shape[1]):
            f.write('dim {:d} homotopy '.format(dim + 1))
            f.write('(from {:.3f} to {:.3f})'.format(z1.cpu().numpy()[0, dim], z2.cpu().numpy()[0, dim]) + '\n')
            for i in range(0, 5):
                f.write(str(i + 1) + '. ')
                z = z1.cpu().numpy()
                z[0, dim] = (1 - 0.25 * i) * z1.cpu().numpy()[0, dim] + 0.25 * i * z2.cpu().numpy()[0, dim]
                z = torch.tensor(z, device=args.device)
                if args.decoding == 'greedy':
                    res = model.decoder.greedy_decoding(z, args.max_length)[0]
                elif args.decoding == 'beam_search':
                    res = model.decoder.beam_search_decoding(z, args.max_length)[0]
                else:
                    raise ValueError('Invalid decoding strategy!')
                if 102 in res:
                    res = res[:res.index(102)]
                f.write(tokenizer.decode(res) + '\n')
            z1 = z
    f.close()


def generation(model, tokenizer, args):
    f = open(os.path.join(os.path.dirname(args.load_model_path), args.decoding + '_generation.txt'), 'w')
    with torch.no_grad():
        for _ in range(50):
            z = torch.randn((200, args.z_dim), device=args.device)
            if args.decoding == 'greedy':
                res = model.decoder.greedy_decoding(z, args.max_length)
            elif args.decoding == 'beam_search':
                res = model.decoder.beam_search_decoding(z, args.max_length)
            else:
                raise ValueError('Invalid decoding strategy!')
            for element in res:
                if 102 in element:
                    element = element[:element.index(102)]
                f.write(tokenizer.decode(element) + '\n')
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--test_path', default='.\\Dataset\\CBT\\test.txt', help='test dataset path')
    parser.add_argument('--load_model_path', default='none', help='training from existing model')
    parser.add_argument('--decoding', default='greedy', help='decoding strategy')
    parser.add_argument('-s', '--seed', default=0, type=int, help='random seed, default: 0')
    args = parser.parse_args()

    if not os.path.exists(args.load_model_path):
        exit(args.load_model_path + ' does not exist.')

    with open(os.path.join(os.path.dirname(args.load_model_path), 'args.json'), 'r', encoding='utf-8') as f:
        config = json.load(f)
    args.emb_dim = config['emb_dim']
    args.lstm_dim = config['lstm_dim']
    args.z_dim = config['z_dim']
    args.batch_size = config['batch_size']
    args.max_length = config['max_length']
    args.is_bert_encoder = config['is_bert_encoder']
    args.bert_type = config['bert_type']
    args.post = config['post']

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.bert_type)
    test_dataset = PlainTextDataset(args.test_path)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    if args.is_bert_encoder:
        model = BERTVAE(args.emb_dim, args.lstm_dim, args.z_dim, len(tokenizer), args.bert_type, args.post)
    else:
        model = LSTMVAE(args.emb_dim, args.lstm_dim, args.z_dim, len(tokenizer), args.post)

    model.load_state_dict(torch.load(args.load_model_path, map_location='cpu'), strict=False)
    model.to(args.device)
    model.eval()

    basic(model, test_dataloader, args)
    get_representations(model, test_dataloader, args)
    reconstruction(model, tokenizer, test_dataloader, args)
    homotopy(model, tokenizer, args)
    generation(model, tokenizer, args)