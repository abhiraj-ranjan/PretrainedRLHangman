#!/usr/bin/env python3
import argparse
import random
import logging
import sys
from collections import deque, namedtuple, Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertForMaskedLM, BertModel, BertTokenizer
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm


# --------------------
# Data utilities
# --------------------
def load_and_split(words_file: str, test_size=0.2, seed=42):
    with open(words_file) as f:
        words = [w.strip() for w in f if w.strip().islower()]
    train, test = train_test_split(words, test_size=test_size, random_state=seed)
    return train, test


class MaskedWordDataset(Dataset):
    """Per-character masked-LM dataset"""

    def __init__(self, words, tokenizer, max_len, mask_prob=0.15):
        self.words = words
        self.tok = tokenizer
        self.max_len = max_len
        self.mask_prob = mask_prob

    def __len__(self):
        return len(self.words)

    def __getitem__(self, i):
        w = self.words[i].upper()
        ids = self.tok.convert_tokens_to_ids(list(w))
        pad = self.tok.pad_token_id
        if len(ids) < self.max_len:
            ids = ids + [pad] * (self.max_len - len(ids))
        else:
            ids = ids[:self.max_len]
        inp, lbl = [], []
        for tid in ids:
            if tid != pad and random.random() < self.mask_prob:
                inp.append(self.tok.mask_token_id);
                lbl.append(tid)
            else:
                inp.append(tid);
                lbl.append(-100)
        return torch.tensor(inp), torch.tensor(lbl)


# --------------------
# Optional head pretraining
# --------------------
def pretrain(args):
    logging.info("Splitting data…")
    train_words, _ = load_and_split(args.words,
                                    test_size=args.test_size,
                                    seed=args.seed)

    tok = BertTokenizer.from_pretrained(args.model_name)
    model = BertForMaskedLM.from_pretrained(args.model_name)
    optim = torch.optim.AdamW(model.parameters(), lr=args.pretrain_lr)

    max_len = max(len(w) for w in train_words)
    loader = DataLoader(
        MaskedWordDataset(train_words, tok, max_len, args.mask_prob),
        batch_size=args.mlm_batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).train()

    step = 0
    pbar = tqdm(total=args.pretrain_steps, desc="Pretrain MLM")
    while step < args.pretrain_steps:
        for inp, lbl in loader:
            inp, lbl = inp.to(device), lbl.to(device)
            out = model(input_ids=inp, labels=lbl)
            out.loss.backward()
            optim.step();
            optim.zero_grad()
            step += 1;
            pbar.update(1)
            if step >= args.pretrain_steps: break
    pbar.close()

    torch.save(model.state_dict(), args.bert_output)
    logging.info(f"Saved head → {args.bert_output}")


# --------------------
# Hangman environment + reward logic (unchanged bonus)
# --------------------
Transition = namedtuple('Transition',
                        ('state', 'mask', 'action', 'reward', 'next_state', 'next_mask', 'done')
                        )


class HangmanEnv:
    def __init__(self, word_list, max_wrong=6):
        self.word_list = word_list
        self.max_wrong = max_wrong
        self.vocab = [chr(i + 97) for i in range(26)] + ['_']
        self.char2idx = {c: i for i, c in enumerate(self.vocab)}
        self.max_len = max(len(w) for w in word_list)
        all_letters = "".join(self.word_list)
        counts = Counter(all_letters)
        total = sum(counts.values())
        self.global_letter_frequencies = {c: counts[c] / total for c in self.vocab}

    def reset(self):
        self.target_word = random.choice(self.word_list)
        self.word_state = ['_'] * len(self.target_word)
        self.tried_letters = set()
        self.lives = self.max_wrong
        return self._get_obs()

    def _get_obs(self):
        idxs = [self.char2idx[c] for c in self.word_state]
        pad = [self.char2idx['_']] * (self.max_len - len(idxs))
        return np.array(idxs + pad, dtype=np.int64)

    def legal_mask(self):
        return np.array([c not in self.tried_letters for c in self.vocab],
                        dtype=np.float32)

    def _calculate_correct_guess_reward(self, letter, scale=10):
        # same bonus logic
        gf = self.global_letter_frequencies.get(letter, 0)
        global_reward = gf * scale
        pattern = "".join(self.word_state)
        candidates = [w for w in self.word_list
                      if len(w) == len(self.target_word)
                      and all(p == '_' or p == c for p, c in zip(pattern, w))]
        counts = Counter("".join(candidates))
        total = sum(counts.values())
        denom = total + len(self.vocab)
        rel = (counts.get(letter, 0) + 1) / denom
        relative_reward = rel * scale
        num_underscores = self.word_state.count('_')
        ratio = num_underscores / len(self.word_state)
        reward = global_reward * ratio + relative_reward * (1 - ratio)
        info = {"global_reward": global_reward,
                "relative_reward": relative_reward}
        return reward, info

    def step(self, action):
        letter = self.vocab[action]
        reward = 0.0
        info = {}
        if letter in self.tried_letters:
            reward = -2.0
            self.lives -= 1
        else:
            self.tried_letters.add(letter)
            if letter in self.target_word:
                bonus, binfo = self._calculate_correct_guess_reward(letter)
                reward += 10.0 + bonus
                for i, ch in enumerate(self.target_word):
                    if ch == letter: self.word_state[i] = letter
                if '_' not in self.word_state:
                    reward += 50.0;
                    info['win'] = True
                    return self._get_obs(), reward, True, info
                info.update(binfo)
            else:
                self.lives -= 1
                reward -= 5.0
                if self.lives <= 0:
                    info['win'] = False
                    return self._get_obs(), reward, True, info
        info.setdefault('win', None)
        return self._get_obs(), reward, False, info


# --------------------
# DQN + BERT policy
# --------------------
class DQN(nn.Module):
    def __init__(self, vocab_size, bert_dim, hidden_dim, model_name):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.head = nn.Sequential(
            nn.Linear(bert_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size)
        )

    def forward(self, x, mask):
        att = (x != self.bert.config.pad_token_id).long()
        out = self.bert(input_ids=x, attention_mask=att).pooler_output
        q = self.head(out)
        return q.masked_fill(mask == 0, -1e9)


class Agent:
    def __init__(self, args, device, vocab_size):
        self.device = device
        self.policy = DQN(vocab_size,
                          args.bert_dim,
                          args.hidden_dim,
                          args.model_name).to(device)
        self.target = DQN(vocab_size,
                          args.bert_dim,
                          args.hidden_dim,
                          args.model_name).to(device)
        self.target.load_state_dict(self.policy.state_dict());
        self.target.eval()
        self.optim = optim.Adam(self.policy.parameters(), lr=args.lr)
        self.memory = deque(maxlen=args.memory_size)
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.eps_start = args.eps_start
        self.eps_end = args.eps_end
        self.eps_decay = args.eps_decay
        self.steps_done = 0

    def select(self, state, mask):
        eps = self.eps_end + (self.eps_start - self.eps_end) * np.exp(-self.steps_done / self.eps_decay)
        self.steps_done += 1
        if random.random() < eps:
            return int(random.choice(np.where(mask)[0]))
        st = torch.tensor([state], device=self.device)
        m = torch.tensor([mask], device=self.device)
        q = self.policy(st, m).squeeze(0).detach().cpu().numpy()
        q[mask == 0] = -np.inf
        return int(np.argmax(q))

    def push(self, *args):
        self.memory.append(Transition(*args))

    def optimize(self):
        if len(self.memory) < self.batch_size: return
        batch = random.sample(self.memory, self.batch_size)
        S, M, A, R, S2, M2, D = zip(*batch)
        S = torch.tensor(np.stack(S), device=self.device)
        M = torch.tensor(np.stack(M), device=self.device)
        A = torch.tensor(A, device=self.device).unsqueeze(1)
        R = torch.tensor(R, device=self.device).unsqueeze(1)
        S2 = torch.tensor(np.stack(S2), device=self.device)
        M2 = torch.tensor(np.stack(M2), device=self.device)
        Dn = torch.tensor([0.0 if d else 1.0 for d in D],
                          device=self.device).unsqueeze(1)
        Q1 = self.policy(S, M).gather(1, A)
        A2 = self.policy(S2, M2).argmax(dim=1, keepdim=True)
        Q2 = self.target(S2, M2).gather(1, A2)
        tgt = R + Dn * self.gamma * Q2
        loss = nn.functional.mse_loss(Q1, tgt.detach())
        self.optim.zero_grad();
        loss.backward();
        self.optim.step()

    def update_target(self):
        self.target.load_state_dict(self.policy.state_dict())


# --------------------
# Train + Eval Hybrid MLM+RL
# --------------------
def train_rl(args):
    train_words, test_words = load_and_split(
        args.words, test_size=args.test_size, seed=args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = len(HangmanEnv(train_words).vocab)
    agent = Agent(args, device, vocab_size)
    # prepare quick MLM
    tok = BertTokenizer.from_pretrained(args.model_name)
    mlm = BertForMaskedLM.from_pretrained(args.model_name).to(device)
    mlm.load_state_dict(agent.policy.bert.state_dict(), strict=False)
    mlm_opt = optim.AdamW(mlm.parameters(), lr=args.pretrain_lr)
    max_len = max(len(w) for w in train_words)
    mlm_loader = iter(DataLoader(
        MaskedWordDataset(train_words, tok, max_len, args.mask_prob),
        batch_size=args.mlm_batch_size, shuffle=True))
    best_acc = 0.0

    for ep in range(1, args.episodes + 1):
        env = HangmanEnv(train_words, max_wrong=args.max_wrong)
        st, mask = env.reset(), env.legal_mask()
        # rule-based first guesses
        for ltr in ['e', 'a']:
            idx = env.char2idx[ltr]
            st, _, done, info = env.step(idx)
            mask = env.legal_mask()
            agent.push(st, mask, idx, 0, st, mask, done)
            agent.optimize()
            if done: break
        total_r = 0.0;
        done = False;
        info = {}
        while not done:
            a = agent.select(st, mask)
            s2, r, done, info = env.step(a)
            m2 = env.legal_mask()
            agent.push(st, mask, a, r, s2, m2, done)
            st, mask = s2, m2;
            total_r += r;
            agent.optimize()
        if ep % args.target_update == 0: agent.update_target()
        # quick MLM warmup
        if ep % args.mlm_interval == 0:
            mlm.train()
            for _ in range(args.mlm_steps):
                try:
                    inp, lbl = next(mlm_loader)
                except StopIteration:
                    mlm_loader = iter(mlm_loader);
                    inp, lbl = next(mlm_loader)
                inp, lbl = inp.to(device), lbl.to(device)
                out = mlm(input_ids=inp, labels=lbl)
                out.loss.backward();
                mlm_opt.step();
                mlm_opt.zero_grad()
            agent.policy.bert.load_state_dict(mlm.state_dict(), strict=False)
        # logs
        if ep % args.log_interval == 0:
            eps = args.eps_end + (args.eps_start - args.eps_end) * np.exp(-agent.steps_done / args.eps_decay)
            print(f"[{ep}/{args.episodes}] "
                  f"{'WIN ' if info.get('win') else 'LOSS'}"
                  f"  R={total_r:.2f}  Eps={eps:.3f}", flush=True)
        if ep % args.eval_interval == 0:
            wins = 0
        
            for _ in test_words:
                e = HangmanEnv(test_words, max_wrong=args.max_wrong)
                s0, m0 = e.reset(), e.legal_mask();
                d = False;
                inf = {}
                while not d:
                    a0 = agent.select(s0, m0)
                    s0, _, d, inf = e.step(a0);
                    m0 = e.legal_mask()
                if inf.get('win'): wins += 1
            acc = wins / len(test_words) * 100
            print(f"→ Eval @ {ep}: Test Acc = {acc:.2f}%", flush=True)
            if acc > best_acc:
                best_acc = acc
                torch.save(agent.policy.state_dict(), args.save)
    print(f"\nDone — best test acc = {best_acc:.2f}%"
          f" (saved to {args.save})", flush=True)


# --------------------
# CLI
# --------------------
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s')
    p = argparse.ArgumentParser("RL-BERT Hangman")
    sub = p.add_subparsers(dest='cmd', required=True)
    # pretrain
    p1 = sub.add_parser('pretrain')
    p1.add_argument('--words', required=True)
    p1.add_argument('--bert-output', required=True)
    p1.add_argument('--model-name', default='prajjwal1/bert-tiny')
    p1.add_argument('--pretrain-steps', type=int, default=20000)
    p1.add_argument('--mlm-batch-size', type=int, default=64)
    p1.add_argument('--mask-prob', type=float, default=0.15)
    p1.add_argument('--pretrain-lr', type=float, default=5e-5)
    p1.add_argument('--test-size', type=float, default=0.2)
    p1.add_argument('--seed', type=int, default=42)
    # train
    p2 = sub.add_parser('train')
    p2.add_argument('--words', required=True)
    p2.add_argument('--save', required=True)
    p2.add_argument('--model-name', default='prajjwal1/bert-tiny')
    p2.add_argument('--bert-dim', type=int, default=128)
    p2.add_argument('--hidden-dim', type=int, default=128)
    p2.add_argument('--episodes', type=int, default=50000)
    p2.add_argument('--batch-size', type=int, default=128)
    p2.add_argument('--lr', type=float, default=1e-3)
    p2.add_argument('--gamma', type=float, default=0.99)
    p2.add_argument('--max-wrong', type=int, default=6)
    p2.add_argument('--memory-size', type=int, default=100000)
    p2.add_argument('--eps-start', type=float, default=1.0)
    p2.add_argument('--eps-end', type=float, default=0.01)
    p2.add_argument('--eps-decay', type=float, default=50000)
    p2.add_argument('--target-update', type=int, default=1000)
    p2.add_argument('--eval-interval', type=int, default=500)
    p2.add_argument('--log-interval', type=int, default=100)
    p2.add_argument('--mlm-interval', type=int, default=1000)
    p2.add_argument('--mlm-steps', type=int, default=20)
    p2.add_argument('--mlm-batch-size', type=int, default=32)
    p2.add_argument('--mask-prob', type=float, default=0.15)
    p2.add_argument('--test-size', type=float, default=0.2)
    p2.add_argument('--seed', type=int, default=42)
    # eval
    p3 = sub.add_parser('eval')
    p3.add_argument('--words', required=True)
    p3.add_argument('--model', required=True)
    p3.add_argument('--max-wrong', type=int, default=6)
    p3.add_argument('--test-size', type=float, default=0.2)
    p3.add_argument('--seed', type=int, default=42)

    args = p.parse_args()
    if args.cmd == 'pretrain':
        pretrain(args)
    elif args.cmd == 'train':
        train_rl(args)
    else:
        raise RuntimeError("Unknown command")
