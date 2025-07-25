import argparse
import os
import json
import jieba
import torch
from tqdm import tqdm 

from fairseq import utils
from fairseq.data import FairseqDataset, data_utils
from fairseq.data.dictionary import Dictionary
from torch.utils.data.dataloader import DataLoader
from transformers import BertConfig, BertModel, BertTokenizer


MODEL_CLASSES = {
    "chinese": (BertConfig, BertModel, BertTokenizer),
}


def move_to_device(sample, device):

  def _move_to_device(tensor):
    return tensor.to(device=device)
  
  return utils.apply_to_sample(_move_to_device, sample)


class RawPair2WordAlignDataset(FairseqDataset):

  def __init__(self, ref_lines, can_lines, tokenizer, vocab):
    assert len(ref_lines) == len(can_lines)
    self.ref_lines = ref_lines
    self.can_lines = can_lines
    self.tokenizer = tokenizer
    self.vocab = vocab
    self.bos_idx = self.vocab.bos()
    self.eos_idx = self.vocab.eos()
    self.bos_word = self.vocab[self.bos_idx]
    self.eos_word = self.vocab[self.eos_idx]

  @property
  def sizes(self):
    return [1] * len(self)
  
  def size(self, index):
    return 1

  def get_indices_and_gid(self, line, offset=0):
    line_indices = []
    token_group_id = []
    words = jieba.lcut(line.strip())
    for gid, word in enumerate(words):
      tokens = self.tokenizer.tokenize(word)
      word2indices = self.tokenizer.convert_tokens_to_ids(tokens)
      for subword_idx in word2indices:
        line_indices.append(subword_idx)
        token_group_id.append(gid + offset)
    return line_indices, token_group_id, words
  
  def __getitem__(self, index):

    ref_line = self.ref_lines[index]
    ref_indices, ref_gid, ref_words = self.get_indices_and_gid(ref_line, offset=0)
    can_line = self.can_lines[index]
    can_indices, can_gid, can_words = self.get_indices_and_gid(can_line, offset=0)

    line_indices = [self.vocab.bos()] + ref_indices + [self.vocab.eos()] + can_indices + [self.vocab.eos()]
    token_group_id = [-1] + ref_gid + [-1] + can_gid + [-1]
    assert len(line_indices) == len(token_group_id)

    ref_fr = 1
    ref_to = ref_fr + len(ref_indices)
    can_fr = ref_to + 1
    can_to = can_fr + len(can_indices)
    assert can_to + 1 == len(line_indices)

    return {
      "tensor": torch.LongTensor(line_indices),
      "token_group": token_group_id,
      "ref_words": ref_words,
      "can_words": can_words,
      "offsets": (ref_fr, ref_to, can_fr, can_to),
    }

  def __len__(self):
    return len(self.ref_lines)

  def collater(self, samples):
    if len(samples) == 0:
      return {}
    ref_fr = [s["offsets"][0] for s in samples]
    ref_to = [s["offsets"][1] for s in samples]
    can_fr = [s["offsets"][2] for s in samples]
    can_to = [s["offsets"][3] for s in samples]

    tensor_samples = [s["tensor"] for s in samples]
    token_groups = [s["token_group"] for s in samples]
    ref_words = [s["ref_words"] for s in samples]
    can_words = [s["can_words"] for s in samples]
    pad_idx = self.vocab.pad()
    eos_idx = self.vocab.eos()
    
    tokens = data_utils.collate_tokens(tensor_samples, pad_idx, eos_idx)
    lengths = torch.LongTensor([s.numel() for s in tensor_samples])
    ntokens = sum(len(s) for s in tensor_samples)


    batch = {
      'nsentences': len(samples),
      'ntokens': ntokens,
      'net_input': {
        'ref_tokens': tokens,
        'ref_lengths': lengths,
      },
      'token_groups': token_groups,
      "ref_words": ref_words,
      "can_words": can_words,
      "offsets": (ref_fr, ref_to, can_fr, can_to),
    }

    return batch


def load_model(args):
  args.model_type = args.model_type.lower()
  config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
  config = config_class.from_pretrained(
    args.config_name if args.config_name else args.model_name_or_path,
    cache_dir=args.cache_dir if args.cache_dir else None,
  )
    
  config.output_hidden_states = True
  config.return_dict = False

  tokenizer = tokenizer_class.from_pretrained(
    args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    cache_dir=args.cache_dir if args.cache_dir else None,
  )
  state_dict = None
  model = model_class.from_pretrained(
    args.model_name_or_path,
    config=config,
    state_dict=state_dict,
  )
  return config, tokenizer, model


def get_dataset(args, vocab, tokenizer):
  with open(args.ref_fn) as fp: ref_lines = [l for l in fp]
  with open(args.can_fn) as fp: can_lines = [l for l in fp]
  dataset = RawPair2WordAlignDataset(ref_lines, can_lines, tokenizer, vocab)
  return dataset


def extract_wa_from_pi_xi(pi, xi):
  m, n = pi.size()
  device = pi.device
  forward = torch.eye(n, device=device)[pi.argmax(dim=1)]
  backward =torch.eye(m, device=device)[xi.argmax(dim=0)]
  inter = forward * backward.transpose(0, 1)
  ret = []
  for i in range(m):
    for j in range(n):
      if inter[i, j].item() > 0:
        ret.append((i, j))
  return ret


def _sinkhorn_iter(S, num_iter=2):
  if num_iter <= 0:
    return S, S
  assert S.dim() == 2
  S[S<=0].fill_(1e-6)
  pi = S
  xi = pi
  for i in range(num_iter):
    pi_sum_over_i = pi.sum(dim=0, keepdim=True)
    xi = pi / pi_sum_over_i
    xi_sum_over_j = xi.sum(dim=1, keepdim=True)
    pi = xi / xi_sum_over_j
  return pi, xi


def sinkhorn(sample, batch_sim, ref_fr, ref_to, can_fr, can_to, num_iter=2):
  pred_wa = []
  for i, sim in enumerate(batch_sim):
    sim_wo_offset = sim[ref_fr[i]: ref_to[i], can_fr[i]: can_to[i]]
    if ref_to[i] - ref_fr[i] <= 0 or can_to[i] - can_fr[i] <= 0:
      print("[W] ref or can len=0")
      pred_wa.append([])
      continue
    pi, xi = _sinkhorn_iter(sim_wo_offset, num_iter)
    pred_wa_i_wo_offset = extract_wa_from_pi_xi(pi, xi)
    pred_wa_i = []
    for ref_idx, can_idx in pred_wa_i_wo_offset:
      pred_wa_i.append((ref_idx + ref_fr[i], can_idx + can_fr[i]))
    pred_wa.append(pred_wa_i)
  
  return pred_wa


def convert_batch_fairseq2hf(net_input):
    tokens = net_input["ref_tokens"]
    lengths = net_input["ref_lengths"]
    _, max_len = tokens.size()
    device = tokens.device
    if max_len > 512:
      attention_mask = (torch.arange(512)[None, :].to(device) < lengths[:, None]).float()
      tokens = tokens[:, :512]
    else:
      attention_mask = (torch.arange(max_len)[None, :].to(device) < lengths[:, None]).float()
    return tokens, attention_mask



def get_wa(args, model, sample):
  model.eval()
  with torch.no_grad():
    tokens, attention_mask = convert_batch_fairseq2hf(sample['net_input'])
  last_layer_outputs, first_token_outputs, all_layer_outputs = model(input_ids=tokens, attention_mask=attention_mask)
  wa_features = all_layer_outputs[args.wa_layer]
  rep = wa_features

  ref_fr, ref_to, can_fr, can_to = sample["offsets"]
  batch_sim = torch.bmm(rep, rep.transpose(1,2))
  wa = sinkhorn(sample, batch_sim, ref_fr, ref_to, can_fr, can_to, num_iter=args.sinkhorn_iter)
  ret_wa = []
  for i, wa_i in enumerate(wa):
    gid = sample["token_groups"][i]
    ret_wa_i = set()
    for a in wa_i:
      ret_wa_i.add((gid[a[0]] + 1, gid[a[1]] + 1))
    ret_wa.append(ret_wa_i)
  return ret_wa

def id_to_word(ref_words, tgt_words, alignments):
    word_alignments = []
    for ref_id, tgt_id in alignments:
        ref_word = ref_words[ref_id-1]
        tgt_word = tgt_words[tgt_id-1]
        word_alignments.append((ref_word, tgt_word))
    return word_alignments

def run(args):
  args.vocab_path = os.path.join(args.model_name_or_path, "vocab.txt")
  vocab = Dictionary.load(args.vocab_path)

  config, tokenizer, model = load_model(args)
  if args.device == torch.device("cuda"):
    model.cuda()
  
  trans_pair = args.trans_pair
  ref, can =  trans_pair.split("-")
  test_model = args.test_model
  test_set_dir = os.path.join(args.test_set_dir, trans_pair)
  args.ref_fn = os.path.join(test_set_dir, "test.%s" % ref)
  args.can_fn = os.path.join(test_set_dir, "test.%s" % can)
  out_json_path = os.path.join(args.alignments_output_dir, f"alignments_{test_model}.jsonl")

  dataset = get_dataset(args, vocab, tokenizer)
  dl = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=dataset.collater)

  print("Transalation pair: %s" % trans_pair)
  for wa_layer in range(13):
    if wa_layer == 8:
      args.wa_layer = wa_layer
      result_wa = []
      for sample in tqdm(dl, desc="Processing samples", unit="sample"):
        sample = move_to_device(sample, args.device)
        batch_wa = get_wa(args, model, sample)
        # batch_wa = get_wa_v2(args, model, sample)
        for i, wa in enumerate(batch_wa):
          ref_words = sample["ref_words"][i]
          can_words = sample["can_words"][i]
          wa = sorted(wa, key=lambda x: x[0])
          result_wa.append({
              "ref_words": ref_words,
              "can_words": can_words,
              "alignments_id": list(wa),
          })
      with open(out_json_path, 'w', encoding='utf-8') as f:
        for record in result_wa:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--test_set_dir", type=str, default="alignment-test-sets")
  parser.add_argument("--alignments_output_dir", type=str, default="output")
  parser.add_argument("--test_model", type=str, default="test_model_name")
  parser.add_argument("--trans_pair", type=str, default="ref-can")
  parser.add_argument("--wa_layer", type=int, default=8)
  parser.add_argument("--sinkhorn_iter", type=int, default=2)
  parser.add_argument("--batch_size", type=int, default=32)
  parser.add_argument('--vocab_path', default="", type=str)
  parser.add_argument(
    "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
  )
  parser.add_argument(
    "--tokenizer_name",
    default="",
    type=str,
    help="Pretrained tokenizer name or path if not the same as model_name",
  )
  parser.add_argument(
    "--cache_dir",
    default="",
    type=str,
    help="Where do you want to store the pre-trained models downloaded from s3",
  )
  parser.add_argument("--model_type", default="chinese", type=str, required=True)
  parser.add_argument("--model_name_or_path", default="AnchiBERT", type=str, required=True,
                      help="Path to pre-trained model")
  args = parser.parse_args()
  args.device = torch.device("cuda")
  run(args)