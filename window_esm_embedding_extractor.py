import argparse
from Bio import SeqIO
from Bio.Seq import Seq
import numpy as np
import torch
import esm
import os
import time

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def get_args():
    parser = argparse.ArgumentParser(description="Genomic Reading-Frame Entropy Translator & Analyzer.\nRun entropy analysis on genomic windows from a FASTA file.")

    # Mandatory arguments
    parser.add_argument('-f','--fasta_path', type=str, help='Path to the input FASTA file.')
    parser.add_argument('-o','--out_dir', type=str, help='Output directory where results will be saved.')
    parser.add_argument('-l','--layer', nargs='+', type=int, default=33, help='ESM2 Layer to extract (default: 33).')

    # Optional arguments
    parser.add_argument('--window_size', type=int, default=2001, help='Size of the sliding window (default: 2001).')
    parser.add_argument('--overlap', type=int, default=999, help='Number of bases the windows should overlap (default: 999).')
    parser.add_argument('--model_name', type=str, default='esm2_t33_650M_UR50D', help='Name of the model to use (default: "esm2_t33_650M_UR50D").')
    args = parser.parse_args()
    return args

def load_model(model_name="esm2_t33_650M_UR50D"):
    if model_name == 'esm2_t6_8M_UR50D':
        model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    elif model_name == 'esm2_t12_35M_UR50D':
        model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
    elif model_name == 'esm2_t30_150M_UR50D':
        model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
    elif model_name == 'esm2_t33_650M_UR50D':
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    elif model_name == 'esm1b_t33_650M_UR50S':
        model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    elif model_name == 'esm2_t36_3B_UR50D':
        model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
    elif model_name == 'esm2_t48_15B_UR50D':
        model, alphabet = esm.pretrained.esm2_t48_15B_UR50D()
    model = model.to(DEVICE)
    esm_alphabet_len = len(alphabet.all_toks)
    esm_alphabet = list("".join(alphabet.all_toks[4:24]))
    ALPHABET = "AFILVMWYDEKRHNQSTMMMMGPC"
    ALPHABET_map = [esm_alphabet.index(a) for a in ALPHABET]
    return model, alphabet, esm_alphabet_len, ALPHABET, ALPHABET_map

def fetch_fasta_str(fasta_path):
    with open(fasta_path, 'r') as file:
        genome_record = SeqIO.read(file, 'fasta')
        return str(genome_record.seq)

def genome_window_selector(full_seq, rev_comp=False, window_size=2001, overlap=1000):
    """
    Splits a sequence into a list of subsequences based on the specified window size and overlap. TO DO: ADD READING FRAMES.

    :param full_seq: str
        The full genomic sequence to be split.
    :param rev_comp: boolean
        Return the sub_sequences in the reverse complement.
    :param window_size: int, optional
        The size of each window (subsequence length).
    :param overlap: int, optional
        The overlap size between consecutive windows.

    :return: list of str
        A list containing the subsequences.
    """
    # Initialize an empty list to hold the subsequences
    sub_sequences = []

    # Calculate the step size, which is how much we move forward after each window
    step_size = window_size - overlap

    # Iterate through the full sequence to generate the subsequences
    for i in range(0, len(full_seq), step_size):
        # Extract the subsequence
        sub_seq = full_seq[i:i + window_size]
        if rev_comp:
            sub_seq = str(Seq(sub_seq).reverse_complement())
        sub_sequences.append(sub_seq)

    return sub_sequences

def get_rep0(seq, layer, model, alphabet, exclude_bos_eos=True):
  x,ln = alphabet.get_batch_converter()([(None,seq)])[-1],len(seq)
  for n,a in enumerate(list(seq)):
    if a == "-":
      x[:,n+1] = alphabet.mask_idx
  if exclude_bos_eos:
    x = x[:,1:-1]
  return model(x.to(DEVICE),repr_layers=[layer])["representations"][layer][0].detach().cpu().numpy()

def get_rep(seq, layer, model, alphabet, exclude_bos_eos=True):
  x,ln = alphabet.get_batch_converter()([(None,seq)])[-1],len(seq)
  for n,a in enumerate(list(seq)):
    if a == "-":
      x[:,n+1] = alphabet.mask_idx
  if exclude_bos_eos:
    x = x[:,1:-1]

  reps = model(x.to(DEVICE),repr_layers=layer)["representations"]
  embeddings_list = []
  for l in layer:
    embedding_l = np.array(reps[l].squeeze().detach().cpu().numpy())
    embeddings_list.append(embedding_l)
  embeddings = np.stack(embeddings_list, axis=0)
  return embeddings

def main():
    print(f'ESM2 Embedding Extractor')
    args = get_args()
    model, alphabet, esm_alphabet_len, ALPHABET, ALPHABET_map = load_model(args.model_name)
    fasta_str = fetch_fasta_str(args.fasta_path)
    seq_windows = genome_window_selector(fasta_str,window_size=args.window_size,overlap=args.overlap)
    print(f'> Extracting Layer(s) {args.layer}')
    print(f' > Total Windows: {len(seq_windows)}')
    os.makedirs(args.out_dir, exist_ok=True)
    t0=time.time()
    for i, DNA_window in enumerate(seq_windows):
        print('-'*50,f'\nWindow {i}, Elapsed Time: {round(time.time()-t0)}')
        DNA_rc_window = str(Seq(DNA_window).reverse_complement())
        ref_window_dict = {'FW':DNA_window,'RC':DNA_rc_window}
        for frame in range(3):
            print(f'>> Computing frame: {frame}')
            for strand in ['FW','RC']:
                print(f'>>> Computing strand: {strand}')
                ref_window = ref_window_dict[strand]
                window = str(Seq(ref_window[frame:]).translate()).replace('*','X')
                #Compute embeddings for each layer
                embedding = get_rep(window, args.layer, model,  alphabet)
                for k,l in enumerate(args.layer):
                    saved_file = os.path.join(args.out_dir, f'{i}_{strand}_{frame}_ESM2_emdedding_L{l}_{args.model_name}.txt')
                    np.savetxt(saved_file,embedding[k])

    print(f'Complete! Compute time: {time.time() - t0:.2f} seconds')

if __name__ == '__main__':
    main()
