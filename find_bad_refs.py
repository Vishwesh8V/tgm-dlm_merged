import numpy as np
import os.path as osp
from nltk.translate.bleu_score import corpus_bleu
from rdkit import RDLogger
from Levenshtein import distance as lev
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit import DataStructs
RDLogger.DisableLog('rdApp.*')
from fcd import get_fcd, load_ref_model, canonical_smiles
import warnings
warnings.filterwarnings('ignore')
def get_smis(filepath):
    print(filepath)
    with open(filepath) as f:
        lines = f.readlines()
    gt_smis= []
    op_smis = []
    for s in lines:
        if len(s)<3:
            continue
        s0,s1 = s.split(' || ')
        s0,s1 = s0.strip().replace('[EOS]','').replace('[SOS]','').replace('[X]','').replace('[XPara]','').replace('[XRing]',''),s1.strip()
        gt_smis.append(s1)
        op_smis.append(s0)
    return gt_smis,op_smis

def evaluate(gt_smis,op_smis):
    references = []
    hypotheses = []
    for i, (gt, out) in enumerate(zip(gt_smis,op_smis)):
        gt_tokens = [c for c in gt]
        out_tokens = [c for c in out]
        references.append([gt_tokens])
        hypotheses.append(out_tokens)
    # BLEU score
    bleu_score = corpus_bleu(references, hypotheses)
    references = []
    hypotheses = []
    levs = []
    num_exact = 0
    bad_mols = 0
    for i, (gt, out) in enumerate(zip(gt_smis,op_smis)):
        hypotheses.append(out)
        references.append(gt)
        try:
            m_out = Chem.MolFromSmiles(out)
            m_gt = Chem.MolFromSmiles(gt)
            if Chem.MolToInchi(m_out) == Chem.MolToInchi(m_gt): num_exact += 1
        except:
            bad_mols += 1
        levs.append(lev(out, gt))
    # Exact matching score
    exact_match_score = num_exact/(i+1)
    # Levenshtein score
    levenshtein_score = np.mean(levs)
    validity_score = 1 - bad_mols/len(gt_smis)
    return bleu_score, exact_match_score, levenshtein_score, validity_score


def fevaluate(gt_smis,op_smis, morgan_r=2):
    outputs = []
    bad_mols = 0
    for n, (gt_smi,ot_smi) in enumerate(zip(gt_smis,op_smis)):
        try:
            gt_m = Chem.MolFromSmiles(gt_smi)
            ot_m = Chem.MolFromSmiles(ot_smi)
            if gt_m is None or ot_m is None: raise ValueError('Bad SMILES')
            outputs.append((gt_m, ot_m))
        except:
            bad_mols += 1
    total = len(outputs) + bad_mols
    validity_score = len(outputs) / total if total > 0 else 0.0

    MACCS_sims = []
    morgan_sims = []
    RDK_sims = []
    enum_list = outputs
    for i, (gt_m, ot_m) in enumerate(enum_list):
        MACCS_sims.append(DataStructs.FingerprintSimilarity(MACCSkeys.GenMACCSKeys(gt_m), MACCSkeys.GenMACCSKeys(ot_m), metric=DataStructs.TanimotoSimilarity))
        RDK_sims.append(DataStructs.FingerprintSimilarity(Chem.RDKFingerprint(gt_m), Chem.RDKFingerprint(ot_m), metric=DataStructs.TanimotoSimilarity))
        morgan_sims.append(DataStructs.TanimotoSimilarity(AllChem.GetMorganFingerprint(gt_m,morgan_r), AllChem.GetMorganFingerprint(ot_m, morgan_r)))

    maccs_sims_score = float(np.mean(MACCS_sims)) if len(MACCS_sims) > 0 else 0.0
    rdk_sims_score = float(np.mean(RDK_sims)) if len(RDK_sims) > 0 else 0.0
    morgan_sims_score = float(np.mean(morgan_sims)) if len(morgan_sims) > 0 else 0.0
    return validity_score, maccs_sims_score, rdk_sims_score, morgan_sims_score

def fcdevaluate(qgt_smis,qop_smis):
    gt_smis = []
    ot_smis = []
    for n, (gt_smi,ot_smi) in enumerate(zip(qgt_smis,qop_smis)):
        if len(ot_smi) == 0: ot_smi = '[]'
        gt_smis.append(gt_smi)
        ot_smis.append(ot_smi)
    model = load_ref_model()
    canon_gt_smis = [w for w in canonical_smiles(gt_smis) if w is not None]
    canon_ot_smis = [w for w in canonical_smiles(ot_smis) if w is not None]
    if len(canon_ot_smis) == 0 or len(canon_gt_smis) == 0:
        return float('nan')
    try:
        fcd_sim_score = get_fcd(canon_gt_smis, canon_ot_smis, model)
    except Exception:
        fcd_sim_score = float('nan')
    return fcd_sim_score

def evaluate_file(filepath):
    gt, op = get_smis(filepath)
    bleu_score, exact_match_score, levenshtein_score, _ = evaluate(gt, op)
    validity_score, maccs_sims_score, rdk_sims_score, morgan_sims_score = fevaluate(gt, op)
    fcd_metric_score = fcdevaluate(gt, op)
    return {
        'BLEU Score': bleu_score,
        'Exact Match Rate': exact_match_score,
        'Levenshtein Dist': levenshtein_score,
        'SMILES Validity': validity_score,
        'MACCS Similarity': maccs_sims_score,
        'RDK Similarity': rdk_sims_score,
        'Morgan Similarity': morgan_sims_score,
        'FCD Metric': fcd_metric_score,
    }

def print_single_results(metrics, title="Evaluation Results"):
    print('='*55)
    print(f' {title}')
    print('='*55)
    for k, v in metrics.items():
        if 'Rate' in k or 'Validity' in k:
            print(f' {k:<20} : {v:.4f} ({v*100:.2f}%)')
        else:
            print(f' {k:<20} : {v:.4f}')
    print('='*55)

if __name__ == '__main__':
    import sys, os, glob, re

    filepath = sys.argv[1] if len(sys.argv) > 1 else '/home/ee/phd/eez248435/tgm-dlm_merged/generation_outputs/sampled_smiles_200k_3108_seed108.txt'
    
    dir_name = os.path.dirname(filepath) or '.'
    base_name = os.path.basename(filepath)
    
    # 1. Discover all matching seed files in the same directory
    # If base_name contains _seed<N>, extract prefix pattern
    seed_match = re.search(r'^(.*?)_seed\d+(\.txt)$', base_name)
    if seed_match:
        prefix = seed_match.group(1)
        ext = seed_match.group(2)
        pattern = os.path.join(dir_name, f"{prefix}_seed*{ext}")
        matched_files = glob.glob(pattern)
    else:
        # Check if there are seed files for this base name (e.g. name.txt -> name_seed*.txt)
        name_no_ext, ext = os.path.splitext(base_name)
        pattern = os.path.join(dir_name, f"{name_no_ext}_seed*{ext}")
        matched_files = glob.glob(pattern)
        if not matched_files and os.path.exists(filepath):
            matched_files = [filepath]

    # Sort files naturally by seed number if available
    def extract_seed(f):
        m = re.search(r'_seed(\d+)', os.path.basename(f))
        return int(m.group(1)) if m else 0

    matched_files = sorted(list(set(matched_files)), key=extract_seed)

    if not matched_files:
        print(f"Error: No files found matching '{filepath}' or pattern '{pattern}'")
        sys.exit(1)

    if len(matched_files) == 1:
        # Single file evaluation
        target_file = matched_files[0]
        print(f"Evaluating output file: {target_file}")
        results = evaluate_file(target_file)
        print_single_results(results, f"Evaluation Results: {os.path.basename(target_file)}")
    else:
        # Multi-seed evaluation
        print("="*95)
        print(f" Found {len(matched_files)} Seed Files for Multi-Seed Evaluation:")
        for mf in matched_files:
            seed_num = extract_seed(mf)
            print(f"   -> [Seed {seed_num:5d}] {mf}")
        print("="*95)

        all_results = []
        seed_labels = []
        for mf in matched_files:
            seed_num = extract_seed(mf)
            seed_labels.append(f"Seed {seed_num}" if seed_num != 0 else os.path.basename(mf)[:10])
            print(f"\n[Evaluating {seed_labels[-1]}] {os.path.basename(mf)}...")
            res = evaluate_file(mf)
            all_results.append(res)
            print_single_results(res, f"Results for {seed_labels[-1]}")

        # Compute Mean and Sample Standard Deviation (Std)
        metric_keys = list(all_results[0].keys())
        summary = {}
        for k in metric_keys:
            vals = [res[k] for res in all_results]
            mean_val = float(np.mean(vals))
            std_val = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            summary[k] = (mean_val, std_val, vals)

        # Print Aggregated Summary Table
        print("\n" + "="*95)
        print(f" MULTI-SEED EVALUATION SUMMARY ({len(matched_files)} Seeds: {[extract_seed(f) for f in matched_files]})")
        print("="*95)
        header = f"{'Metric':<22} | {'Mean ± Std':<22} | " + " | ".join([f"{lbl:<12}" for lbl in seed_labels])
        print(header)
        print("-" * len(header))

        for k, (m, s, vals) in summary.items():
            val_strs = " | ".join([f"{v:<12.4f}" for v in vals])
            print(f"{k:<22} | {m:8.4f} ± {s:<10.4f} | {val_strs}")
        print("="*95)