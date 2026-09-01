from mydatasets import get_dataloader,ChEBIdataset
import torch
import transformers
from mytokenizers import SimpleSmilesTokenizer,regexTokenizer
from transformers import AutoModel
from transformers import AutoTokenizer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i","--input",required=True)
parser.add_argument("--dataset_dir",default='../../datasets/SMILES/',
                    help="Directory containing {split}.txt, in the same layout as datasets/SMILES/. "
                    "Point this at a different dataset (e.g. ../../datasets/RETRO/) to precompute "
                    "SciBERT states for that dataset's condition column instead of ChEBI descriptions.")
args = parser.parse_args()
split = args.input
smtokenizer = regexTokenizer(path=args.dataset_dir.rstrip('/') + '/generate_vocab.txt')
train_dataset = ChEBIdataset(
        dir=args.dataset_dir,
        smi_tokenizer=smtokenizer,
        split=split,
        replace_desc=False,
        load_state=False
        # pre = pre
    )
model = AutoModel.from_pretrained('../../scibert')
tokz = AutoTokenizer.from_pretrained('../../scibert')

volume = {}


model = model.cuda()
    # alllen = []
model.eval()
with torch.no_grad():
    for i in range(len(train_dataset)):
        if i%190 == 0:
            print(i)
        id = train_dataset[i]['cid']
        desc =train_dataset[i]['desc']
        tok_op = tokz(
            desc,max_length=216, truncation=True,padding='max_length'
            )
        toked_desc = torch.tensor(tok_op['input_ids']).unsqueeze(0)
        toked_desc_attentionmask = torch.tensor(tok_op['attention_mask']).unsqueeze(0)
        assert(toked_desc.shape[1]==216)
        lh = model(toked_desc.cuda()).last_hidden_state
        volume[id] = {'states':lh.to('cpu'),'mask':toked_desc_attentionmask}



torch.save(volume,args.dataset_dir.rstrip('/') + '/' + split+'_desc_states.pt')
