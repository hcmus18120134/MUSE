# /content/MUSE/dumped/debug/4sfyvayqsv/best_mapping.pth
# /content/MUSE/dumped/debug/4sfyvayqsv/debug/z2atp7p47y/best_mapping.pth
import os
from collections import OrderedDict

from src.utils import bool_flag, initialize_exp
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
parser.add_argument("--src_emb", type=str, default="", help="Reload source embeddings")
parser.add_argument("--tgt_emb", type=str, default="", help="Reload target embeddings")
parser.add_argument("--max_vocab", type=int, default=-1, help="Maximum vocabulary size (-1 to disable)")
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
trainer = Trainer(src_emb, tgt_emb, mapping, None, params)
trainer.reload_best("/content/MUSE/dumped/debug/4sfyvayqsv/best_mapping.pth")
evaluator = Evaluator(trainer)


# # run evaluations
# to_log = OrderedDict({'n_iter': 0})
# evaluator.monolingual_wordsim(to_log)
# # evaluator.monolingual_wordanalogy(to_log)
# if params.tgt_lang:
#     evaluator.crosslingual_wordsim(to_log)
#     evaluator.word_translation(to_log)
#     evaluator.sent_translation(to_log)