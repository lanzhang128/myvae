import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel


class LSTMEncoder(nn.Module):
    def __init__(self, embedding, lstm_dim, emb_dim, z_dim, post='diag'):
        super().__init__()
        self.post = post
        self.encoder_emb = embedding
        self.encoder_lstm = nn.LSTM(input_size=emb_dim, hidden_size=lstm_dim, batch_first=True)
        self.encoder_mean_layer = nn.Linear(lstm_dim, z_dim)
        if self.post == 'diag':
            self.encoder_logvar_layer = nn.Linear(lstm_dim, z_dim)
        elif self.post == 'iso':
            self.encoder_logvar_layer = nn.Linear(lstm_dim, 1)
        else:
            raise ValueError('Not a valid posterior.')

    def forward(self, x):
        embeddings = self.encoder_emb(x)
        outputs = self.encoder_lstm(embeddings)[0]
        outputs = torch.sum(x.eq(102).unsqueeze(dim=2) * outputs, dim=1)
        mean = self.encoder_mean_layer(outputs)
        if self.post == 'diag':
            logvar = self.encoder_logvar_layer(outputs)
        else:
            logvar = self.encoder_logvar_layer(outputs)
            logvar = torch.repeat_interleave(logvar, mean.shape[1], dim=1)
        epsilon = torch.randn(mean.shape, device=mean.device)
        z = mean + torch.exp(0.5 * logvar) * epsilon
        kld = 0.5 * torch.sum(torch.square(mean) + torch.exp(logvar) - 1 - logvar, dim=1)
        return z, mean, logvar, kld


class BERTEncoder(nn.Module):
    def __init__(self, z_dim, pretrain_path=None, post='diag'):
        super().__init__()
        self.bert_model = AutoModel.from_pretrained(pretrain_path)
        self.bert_model.train()
        self.post = post
        self.encoder_mean_layer = nn.Linear(self.bert_model.config.hidden_size, z_dim)
        if self.post == 'diag':
            self.encoder_logvar_layer = nn.Linear(self.bert_model.config.hidden_size, z_dim)
        elif self.post == 'iso':
            self.encoder_logvar_layer = nn.Linear(self.bert_model.config.hidden_size, 1)
        else:
            raise ValueError('Not a valid posterior.')

    def forward(self, x):
        outputs = self.bert_model(**x)['last_hidden_state'][:, 0, :]
        mean = self.encoder_mean_layer(outputs)
        if self.post == 'diag':
            logvar = self.encoder_logvar_layer(outputs)
        else:
            logvar = self.encoder_logvar_layer(outputs)
            logvar = torch.repeat_interleave(logvar, mean.shape[1], dim=1)
        epsilon = torch.randn(mean.shape, device=mean.device)
        z = mean + torch.exp(0.5 * logvar) * epsilon
        kld = 0.5 * torch.sum(torch.square(mean) + torch.exp(logvar) - 1 - logvar, dim=1)
        return z, mean, logvar, kld


class BeamSearchNode:
    def __init__(self, tokens, state, log_prob):
        self.tokens = tokens
        self.state = state
        self.log_prob = log_prob


class LSTMDecoder(nn.Module):
    def __init__(self, embedding, lstm_dim, emb_dim, z_dim, vocab_size):
        super().__init__()
        self.decoder_emb = embedding
        self.decoder_lstm = nn.LSTM(input_size=emb_dim+z_dim, hidden_size=lstm_dim, batch_first=True)
        self.vocab_prob = nn.Linear(lstm_dim, vocab_size)

    def forward(self, x, z):
        embeddings = self.decoder_emb(x[:, :-1])
        new_z = torch.repeat_interleave(z.unsqueeze(dim=1), embeddings.shape[1], dim=1)
        inputs = torch.cat([embeddings, new_z], dim=2)
        outputs = self.decoder_lstm(inputs)[0]
        logits = self.vocab_prob(outputs)
        y = x[:, 1:]
        rec = F.cross_entropy(logits.permute((0, 2, 1)), y, reduction='none') * y.ne(0)
        return F.softmax(logits, dim=2), torch.sum(rec, dim=1)

    def greedy_decoding(self, z, maxlen):
        y = 101 * torch.ones((z.shape[0], 1), dtype=torch.int64, device=z.device)
        state = None
        res = torch.ones((z.shape[0], 0), dtype=torch.int64, device=z.device)
        for _ in range(0, maxlen - 1):
            embeddings = self.decoder_emb(y)
            new_z = torch.repeat_interleave(z.unsqueeze(dim=1), embeddings.shape[1], dim=1)
            inputs = torch.cat([embeddings, new_z], dim=2)
            if state is None:
                outputs, state = self.decoder_lstm(inputs)
            else:
                outputs, state = self.decoder_lstm(inputs, state)
            probs = F.softmax(self.vocab_prob(outputs), dim=2)
            y = torch.argmax(probs, dim=2)
            res = torch.cat([res, y], dim=1)
        return res.cpu().numpy().tolist()

    def beam_search_decoding(self, z, maxlen, size=5):
        res = []
        for index in range(z.shape[0]):
            temp_z = z[index:index+1]
            state = None
            hypotheses = [BeamSearchNode([101], state, 0.0)]
            completed = []
            count = 0
            while len(completed) < size and count < maxlen - 1:
                count += 1
                words = []
                hs = []
                cs = []
                pre_log_prob = []
                for node in hypotheses:
                    words.append(node.tokens[-1:])
                    if node.state is not None:
                        hs.append(node.state[0])
                        cs.append(node.state[1])
                    pre_log_prob.append([node.log_prob])
                embeddings = self.decoder_emb(torch.tensor(words, dtype=torch.int64, device=z.device))
                new_z = torch.repeat_interleave(temp_z.unsqueeze(dim=1), embeddings.shape[0], dim=0)
                inputs = torch.cat([embeddings, new_z], dim=2)
                if hs:
                    state = (torch.cat(hs, dim=1), torch.cat(cs, dim=1))
                else:
                    state = None
                outputs, state = self.decoder_lstm(inputs, state)
                log_probs = torch.log(F.softmax(self.vocab_prob(outputs), dim=2))
                log_probs = log_probs + torch.tensor(pre_log_prob, device=log_probs.device).unsqueeze(2)
                new_log_porbs, indices = torch.topk(log_probs.view(-1), size - len(completed))
                new_hypotheses = []
                for i in range(indices.shape[0]):
                    node_id = indices[i].item() // log_probs.shape[2]
                    word_id = indices[i].item() % log_probs.shape[2]
                    new_node = BeamSearchNode(hypotheses[node_id].tokens + [word_id],
                                              (state[0][:, node_id:node_id+1, :], state[1][:, node_id:node_id+1, :]),
                                              new_log_porbs[i].item())
                    if word_id == 102:
                        completed.append(new_node)
                    else:
                        new_hypotheses.append(new_node)
                hypotheses = new_hypotheses
            if count == maxlen - 1:
                completed += hypotheses
            res_id = 0
            for i in range(1, len(completed)):
                if completed[i].log_prob > completed[res_id].log_prob:
                    res_id = i
            res.append(completed[res_id].tokens[1:])
        return res


class LSTMVAE(nn.Module):
    def __init__(self, emb_dim, lstm_dim, z_dim, vocab_size, post='diag'):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim, padding_idx=0)
        self.encoder = LSTMEncoder(self.emb, lstm_dim, emb_dim, z_dim, post)
        self.decoder = LSTMDecoder(self.emb, lstm_dim, emb_dim, z_dim, vocab_size)

    def forward(self, x, beta=1.0, C=0):
        z, mean, logvar, kld = self.encoder(x)
        probs, rec = self.decoder(x, z)
        loss = rec + beta * torch.abs(kld - C)
        return torch.mean(loss, dim=0), torch.mean(kld, dim=0), torch.mean(rec, dim=0)


class BERTVAE(nn.Module):
    def __init__(self, emb_dim, lstm_dim, z_dim, vocab_size, pretrain_path=None, post='diag'):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim, padding_idx=0)
        self.encoder = BERTEncoder(z_dim, pretrain_path, post)
        self.decoder = LSTMDecoder(self.emb, lstm_dim, emb_dim, z_dim, vocab_size)

    def forward(self, x, beta=1.0, C=0):
        z, mean, logvar, kld = self.encoder(x)
        probs, rec = self.decoder(x['input_ids'], z)
        loss = rec + beta * torch.abs(kld - C)
        return torch.mean(loss, dim=0), torch.mean(kld, dim=0), torch.mean(rec, dim=0)