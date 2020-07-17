# /content/MUSE/dumped/debug/4sfyvayqsv/best_mapping.pth
# /content/MUSE/dumped/debug/4sfyvayqsv/debug/z2atp7p47y/best_mapping.pth
import argparse
import os
from collections import OrderedDict

from src.utils import bool_flag, initialize_exp, load_embeddings
from src.models import build_model
from src.trainer import Trainer
from src.evaluation import Evaluator

parser = argparse.ArgumentParser(description='Evaluation')
parser.add_argument("--verbose", type=int, default=2, help="Verbose level (2:debug, 1:info, 0:warning)")
parser.add_argument("--exp_path", type=str, default="", help="Where to store experiment logs and models")
parser.add_argument("--exp_name", type=str, default="debug", help="Experiment name")
parser.add_argument("--exp_id", type=str, default="", help="Experiment ID")
parser.add_argument("--cuda", type=bool_flag, default=True, help="Run on GPU")
# data
parser.add_argument("--src_lang", type=str, default="zh", help="Source language")
parser.add_argument("--tgt_lang", type=str, default="en", help="Target language")
parser.add_argument("--dico_eval", type=str, default="default", help="Path to evaluation dictionary")
# reload pre-trained embeddings
parser.add_argument("--src_emb", type=str, default="/content/MUSE/dumped/debug/chom1xyupn/vectors-zh.txt", help="Reload source embeddings")
parser.add_argument("--tgt_emb", type=str, default="/content/MUSE/dumped/debug/chom1xyupn/vectors-en.txt", help="Reload target embeddings")
parser.add_argument("--max_vocab", type=int, default=100000, help="Maximum vocabulary size (-1 to disable)")
parser.add_argument("--emb_dim", type=int, default=500, help="Embedding dimension")
parser.add_argument("--normalize_embeddings", type=str, default="center", help="Normalize embeddings before training")


# parse parameters
params = parser.parse_args()

# check parameters
assert params.src_lang, "source language undefined"
assert os.path.isfile(params.src_emb)
assert not params.tgt_lang or os.path.isfile(params.tgt_emb)
assert params.dico_eval == 'default' or os.path.isfile(params.dico_eval)

# build logger / model / trainer / evaluator
logger = initialize_exp(params)
src_emb, tgt_emb, mapping, _ = build_model(params, False)

# sau buoc nay se co params.src_dico, params.tgt_dico

trainer = Trainer(src_emb, tgt_emb, mapping, None, params)

# reload mapping
trainer.reload_best("/content/MUSE/dumped/debug/chom1xyupn/best_mapping.pth") 

def translate_word(src_word: str) -> str:
    global src_emb, tgt_emb

    src_word2id = params.src_dico.word2id
    src_lang = params.src_dico.lang
    tgt_word2id = params.tgt_dico.word2id
    tgt_lang = params.tgt_dico.lang


    # mapped word embeddings
    src_emb_clone = src_emb.weight.data
    tgt_emb_clone = tgt_emb.weight.data
    src_word_vec = dict([(w, src_emb_clone[src_word2id[w]]) for w in src_word2id])
    tgt_word_vec = dict([(w, tgt_emb_clone[tgt_word2id[w]]) for w in tgt_word2id])
    word_vect = {src_lang: src_word_vec, tgt_lang: tgt_word_vec}

    lg_keys = src_lang # zh
    lg_query = tgt_lang # en


    #####
    '''
    x: word in src lang
    x -> id_x -> emb[id_x] -> dung nn so voi tung vector cua words trong tgt_lang -> top k -> return
    '''
    src_emb_vec = src_word_vec[src_word].unsqueeze(0) # tensor: [1, embed_dim]
    best_scores = -1
    for tgt_word, tgt_emb_vec in zip(tgt_word_vec.keys(), tgt_word_vec.values()):
        score = src_emb_vec.mm(tgt_emb_vec.unsqueeze(0).transpose(0, 1)) # [1, 1]
        score = score.squeeze(0).squeeze(0).item()
        if score > best_scores:
            best_scores = score
            best_word = tgt_word

    return best_word

print(translate_word('恐龍裝'))
print(translate_word('奢華金'))
print(translate_word('俱家電'))


# # run evaluations
# # evaluator.monolingual_wordanalogy(to_log)
# if params.tgt_lang:
#     evaluator.crosslingual_wordsim(to_log)
#     evaluator.sent_translation(to_log)