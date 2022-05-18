import os
import logging
import time
import json
import argparse
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim import Adam
from transformers import AutoTokenizer
from dataset import PlainTextDataset
from modeling import LSTMVAE, BERTVAE


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--train_path', default='.\\Dataset\\DBpedia\\train.txt', help='training dataset path')
    parser.add_argument('--val_path', default='.\\Dataset\\DBpedia\\valid.txt', help='validation dataset path')
    parser.add_argument('--post', default='diag', help='posterior')
    parser.add_argument('--model_type', default='beta', help='model type')
    parser.add_argument('--model_save_dirpath', default='.\\model\\test', help='model directory')
    parser.add_argument('--load_model_path', default='none', help='training from existing model')
    parser.add_argument('-beta', default=1, type=float, help='beta for training VAE, default: 1')
    parser.add_argument('-C', default=0, type=float, help='C for training VAE, default: 0')
    parser.add_argument('--cycle', default=4, type=int, help='cyclical times, default: 4')
    parser.add_argument('--rate', default=0.5, type=float, help='rate, default: 0.5')
    parser.add_argument('--num', default=0, type=int, help='the number of steps after which recording dimensional KL,'
                                                           ' if num=0, then do not record, default: 0')
    parser.add_argument('-s', '--seed', default=0, type=int, help='random seed, default: 0')
    args = parser.parse_args()

    with open('config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    args.emb_dim = config['emb_dim']
    args.lstm_dim = config['lstm_dim']
    args.z_dim = config['z_dim']
    args.epochs = config['epochs']
    args.batch_size = config['batch_size']
    args.lr = config['lr']
    args.max_length = config['max_length']
    args.is_bert_encoder = config['is_bert_encoder']
    args.bert_type = config['bert_type']

    if not os.path.exists(args.model_save_dirpath):
        os.system('mkdir ' + args.model_save_dirpath)
    if not os.path.exists(os.path.join(args.model_save_dirpath, 'args.json')):
        with open(os.path.join(args.model_save_dirpath, 'args.json'), 'w', encoding='utf-8') as f:
            json.dump(args.__dict__, f, ensure_ascii=False, indent=4)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    log_name = os.path.join(args.model_save_dirpath, time.strftime('%Y%m%d%H%M', time.localtime(time.time())) + '.log')
    file_handler = logging.FileHandler(log_name, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s: %(message)s'))
    logger.addHandler(file_handler)
    logger.info(
        f'training configuration:\nepochs: {args.epochs:<d}\nbatch size: {args.batch_size:<d}\nlr: {args.lr:<8.6f}\n'
        f'max length: {args.max_length}\ntraining dataset: {args.train_path}\nvalidation dataset: {args.val_path}\n'
        f'model save path: {args.model_save_dirpath}\n')

    if args.model_type == 'beta':
        logger.info(f'model type: {args.model_type} with beta={args.beta:<.2f}.\n')
    elif args.model_type == 'cci':
        logger.info(f'model type: {args.model_type} with C={args.C:<.2f}.\n')
    elif args.model_type == 'cyclical':
        logger.info(f'model type: {args.model_type} with cycle={args.cycle:<d} and rate={args.rate:<.2f}.\n')
    elif args.model_type == 'capacity':
        logger.info(f'model type: {args.model_type} with C={args.C:<.2f} finally.\n')
    else:
        raise ValueError(f'Not support model type {args.model_type}!')

    torch.manual_seed(args.seed)

    train_dataset = PlainTextDataset(args.train_path)
    val_dataset = PlainTextDataset(args.val_path)

    logger.info(f'training set has {len(train_dataset)} sentences')
    logger.info(f'validation set has {len(val_dataset)} sentences')

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    tokenizer = AutoTokenizer.from_pretrained(args.bert_type)
    if args.is_bert_encoder:
        model = BERTVAE(args.emb_dim, args.lstm_dim, args.z_dim, len(tokenizer), args.bert_type, args.post)
    else:
        model = LSTMVAE(args.emb_dim, args.lstm_dim, args.z_dim, len(tokenizer), args.post)

    if os.path.exists(args.load_model_path):
        logger.info(f'training on model from {args.load_model_path}')
        model.load_state_dict(torch.load(args.load_model_path, map_location='cpu'), strict=False)
    else:
        logger.info('training from beginning')
    model.to(args.device)

    optimizer = Adam(model.parameters(), lr=args.lr)

    total_step = args.epochs * len(train_dataloader)
    annealing_step = total_step // args.cycle
    if total_step % args.cycle != 0:
        annealing_step += 1
    step_count = 0
    if args.num > 0:
        dim_kl = torch.zeros(size=(0, args.z_dim))
        recs = torch.zeros(size=(0, 1))
        steps = np.zeros(shape=(0, 1))
        dim_kl_record = torch.zeros(size=(0, args.z_dim))
        rec_record = torch.zeros(size=(0, 1))

    for epoch in range(args.epochs):
        print('================================  Train  ================================')
        print('Epoch ' + str(epoch + 1))
        print('============================================================================')
        print('Train Error:')
        logger.debug('================================  Train  ================================')
        logger.debug('Epoch ' + str(epoch + 1))
        logger.debug('============================================================================')
        logger.debug('Train Error:')
        model.train()
        total_loss = 0
        kld_loss = 0
        rec_loss = 0
        for step, batch_data in enumerate(train_dataloader):
            step_count += 1
            if args.is_bert_encoder:
                x = tokenizer(batch_data, max_length=args.max_length, padding='max_length', truncation=True,
                              return_tensors='pt')
                x = x.to(args.device)
            else:
                x = tokenizer(batch_data, max_length=args.max_length, padding='max_length', truncation=True,
                              return_token_type_ids=False, return_attention_mask=False, return_tensors='pt')
                x = x['input_ids'].to(args.device)
            if args.model_type == 'beta':
                loss, kld, rec = model(x, args.beta)
            elif args.model_type == 'cci':
                if step_count <= 1000:
                    C_value = args.C * step_count / 1000
                else:
                    C_value = args.C
                loss, kld, rec = model(x, 1.0, C_value)
            elif args.model_type == 'cyclical':
                temp = ((step_count - 1) % annealing_step) / annealing_step
                if temp < args.rate:
                    beta = temp / args.rate
                else:
                    beta = 1.0
                loss, kld, rec = model(x, beta)
            else:
                C_value = args.C * step_count / total_step
                loss, kld, rec = model(x, 1.0, C_value)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if args.num > 0:
                model.eval()
                with torch.no_grad():
                    z, mean, logvar, _ = model.encoder(x)
                    if args.is_bert_encoder:
                        _, temp_rec = model.decoder(x['input_ids'], z)
                    else:
                        _, temp_rec = model.decoder(x, z)
                    temp_dim_kl = 0.5 * (torch.square(mean) + torch.exp(logvar) - 1 - logvar)
                    dim_kl_record = torch.cat([dim_kl_record, torch.mean(temp_dim_kl.cpu(), dim=0).unsqueeze(0)], dim=0)
                    rec_record = torch.cat([rec_record, torch.mean(temp_rec.cpu(), dim=0).unsqueeze(0).unsqueeze(0)], dim=0)
                model.train()

                if step_count % args.num == 0:
                    dim_kl = torch.cat([dim_kl, torch.mean(dim_kl_record, dim=0).unsqueeze(0)], dim=0)
                    recs = torch.cat([recs, torch.mean(rec_record, dim=0).unsqueeze(0)], dim=0)
                    steps = np.concatenate([steps, step_count * np.ones(shape=(1, 1))], axis=0)
                    dim_kl_record = torch.zeros(size=(0, args.z_dim))
                    rec_record = torch.zeros(size=(0, 1))

            total_loss += loss.item()
            kld_loss += kld.item()
            rec_loss += rec.item()
            if (step + 1) % 50 == 0:
                print(f'step: {step + 1}, loss: {loss.item():<6.4f}, kld: {kld.item():<6.4f}, rec: {rec.item():<6.4f}')
                logger.debug(f'step: {step + 1}, loss: {loss.item():<6.4f}, kld: {kld.item():<6.4f}, rec: {rec.item():<6.4f}')

        print(f'Avg: loss: {total_loss / (step + 1):<8.4f}, kld: {kld_loss / (step + 1):<8.4f}, rec: {rec_loss / (step + 1):<8.4f}')
        logger.debug(f'Avg: loss: {total_loss / (step + 1):<8.4f}, kld: {kld_loss / (step + 1):<8.4f}, rec: {rec_loss / (step + 1):<8.4f}')

        print('================================  Evaluate  ================================')
        logger.info('================================  Evaluate  ================================')
        model.eval()
        total_loss = 0
        kld_loss = 0
        rec_loss = 0
        with torch.no_grad():
            for step, batch_data in enumerate(val_dataloader):
                if args.is_bert_encoder:
                    x = tokenizer(batch_data, max_length=args.max_length, padding='max_length', truncation=True,
                                  return_tensors='pt')
                    x = x.to(args.device)
                else:
                    x = tokenizer(batch_data, max_length=args.max_length, padding='max_length', truncation=True,
                                  return_token_type_ids=False, return_attention_mask=False, return_tensors='pt')
                    x = x['input_ids'].to(args.device)
                loss, kld, rec = model(x)
                total_loss += loss.item()
                kld_loss += kld.item()
                rec_loss += rec.item()
        print(f'Avg: loss: {total_loss / (step + 1):<8.4f}, kld: {kld_loss / (step + 1):<8.4f}, rec: {rec_loss / (step + 1):<8.4f}')
        logger.debug(f'Avg: loss: {total_loss / (step + 1):<8.4f}, kld: {kld_loss / (step + 1):<8.4f}, rec: {rec_loss / (step + 1):<8.4f}')

    save_path = os.path.join(args.model_save_dirpath, f'weights.pth')
    torch.save(model.state_dict(), save_path)
    logger.debug('Save model at: ' + save_path)

    if args.num > 0:
        recs = recs.numpy()
        dim_kl = dim_kl.numpy()

        capacity_data = np.concatenate([steps, recs, dim_kl], axis=-1)
        df = pd.DataFrame(capacity_data)
        df.columns = ['step', 'rec'] + ['kl'+str(i) for i in range(1, args.z_dim+1)]
        df.to_csv(os.path.join(args.model_save_dirpath, 'capacity.csv'), index=False)