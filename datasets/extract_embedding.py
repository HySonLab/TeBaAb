import argparse
import pickle
from pathlib import Path
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, BertModel, BertTokenizer
from esm.models.esmc import ESMC
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer


def get_device(device_str: str) -> torch.device:
    """Validates and returns a torch.device object."""
    if torch.cuda.is_available() and device_str.startswith("cuda"):
        try:
            return torch.device(device_str)
        except Exception as e:
            print(f"Warning: Could not use {device_str}, falling back to 'cuda:0'. Error: {e}")
            return torch.device("cuda:0")
    elif device_str == "cpu":
        return torch.device("cpu")
    else:
        print("Warning: CUDA not available or invalid device specified. Falling back to CPU.")
        return torch.device("cpu")


def get_antigen_emb(df: pd.DataFrame, args: argparse.Namespace):
    """Generates and saves antigen embeddings using ESMC."""
    print("Generating Antigen (ESMC) embeddings...")
    device = get_device(args.device)

    tokenizer = EsmSequenceTokenizer()
    model = ESMC(
        d_model=960,
        n_heads=30,
        n_layers=15,
        use_flash_attn=True,
        tokenizer=tokenizer
    ).eval().to(device)

    try:
        state_dict = torch.load(args.esmc_cache, map_location=device)
        model.load_state_dict(state_dict, strict=False)
    except FileNotFoundError:
        print(f"Error: ESMC model file not found at {args.esmc_cache}")
        return
    except Exception as e:
        print(f"Error loading ESMC model: {e}")
        return

    antigens = df['antigen'].unique()  # Use unique to avoid re-computing
    embeddings = {}

    for ag in tqdm(antigens, desc="Antigen Embeddings"):
        if pd.isna(ag):
            continue
        with torch.no_grad():
            batch = tokenizer.batch_encode_plus(
                [ag],
                padding="max_length",
                truncation=True,
                max_length=256,
                return_tensors="pt",
                return_attention_mask=True,
            )

            try:
                results = model(
                    sequence_tokens=batch["input_ids"].to(device),
                    sequence_id=batch["attention_mask"].to(device),
                )

                if args.embedding_type == "pooler":
                    mask = batch["attention_mask"].unsqueeze(-1).expand_as(results.embeddings).to(device)
                    emb = (results.embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                else:  # "sequence"
                    emb = results.embeddings  # Shape: [1, seq_len, 960]

                embeddings[ag] = emb[0].cpu()
                del emb, batch, results
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error processing antigen '{ag}': {e}")
                continue

    output_file = args.output_dir / f"{args.output_prefix}_antigen.pkl"
    with open(output_file, "wb") as f:
        pickle.dump(embeddings, f)
    print(f"Antigen embeddings saved to {output_file}")


def get_ab_emb(df: pd.DataFrame, args: argparse.Namespace):
    """Generates and saves antibody embeddings using IgBert."""
    print("Generating Antibody (IgBert) embeddings...")
    device = get_device(args.device)

    try:
        tokenizer = BertTokenizer.from_pretrained(args.igbert_cache, do_lower_case=False)
        model = BertModel.from_pretrained(args.igbert_cache, add_pooling_layer=False).to(device).eval()
    except Exception as e:
        print(f"Error loading IgBert model from {args.igbert_cache}: {e}")
        print("Please ensure 'Exscientia/IgBert' is downloaded or the path is correct.")
        return

    df_subset = df[['heavy_chain', 'light_chain']].drop_duplicates()
    heavy_chains = df_subset['heavy_chain'].values
    light_chains = df_subset['light_chain'].values

    paired_sequences = [
        ' '.join(h) + '|' + ' '.join(l)
        for h, l in zip(heavy_chains, light_chains)
        if pd.notna(h) and pd.notna(l)
    ]

    # Create a map back to the original (h, l) tuples
    valid_pairs = [
        (h, l) for h, l in zip(heavy_chains, light_chains)
        if pd.notna(h) and pd.notna(l)
    ]

    embeddings = {}
    for i, seq in tqdm(enumerate(paired_sequences), desc="Antibody Embeddings", total=len(paired_sequences)):
        with torch.no_grad():
            try:
                tokens = tokenizer.batch_encode_plus(
                    [seq],
                    add_special_tokens=True,
                    padding="max_length",  # Use padding="max_length"
                    max_length=512,  # Specify max_length
                    truncation=True,
                    return_tensors="pt",
                    return_special_tokens_mask=True
                )

                output = model(
                    input_ids=tokens['input_ids'].to(device),
                    attention_mask=tokens['attention_mask'].to(device)
                )

                residue_embeddings = output.last_hidden_state.cpu()

                if args.embedding_type == "pooler":
                    # Mask out special tokens for pooling
                    residue_embeddings[tokens["special_tokens_mask"] == 1] = 0
                    sequence_embeddings_sum = residue_embeddings.sum(1)
                    # Count non-special tokens
                    sequence_lengths = torch.sum(tokens["special_tokens_mask"] == 0, dim=1)
                    valid_residue_embeddings = sequence_embeddings_sum / sequence_lengths.unsqueeze(1).clamp(min=1)
                else:  # "sequence"
                    # Return all non-special-token embeddings
                    valid_residue_embeddings = residue_embeddings[tokens["special_tokens_mask"] == 0]

                h, l = valid_pairs[i]
                embeddings[(h, l)] = valid_residue_embeddings[0]

                del tokens, output, residue_embeddings, valid_residue_embeddings
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error processing antibody pair '{valid_pairs[i]}': {e}")
                continue

    output_file = args.output_dir / f"{args.output_prefix}_antibody.pkl"
    with open(output_file, "wb") as f:
        pickle.dump(embeddings, f)
    print(f"Antibody embeddings saved to {output_file}")


def get_text_emb(df: pd.DataFrame, args: argparse.Namespace):
    """Generates and saves text description embeddings using SciBert."""
    print("Generating Text (SciBert) embeddings...")
    device = get_device(args.device)

    try:
        tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased", cache_dir=args.scibert_cache)
        model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased", cache_dir=args.scibert_cache).to(device).eval()
    except Exception as e:
        print(f"Error loading SciBert model from {args.scibert_cache}: {e}")
        print("Please ensure 'allenai/scibert_scivocab_uncased' is downloaded or the path is correct.")
        return

    # Use pdb_id as the key, assuming one description per pdb_id
    df_subset = df[['pdb_id', 'description']].drop_duplicates(subset=['pdb_id'])
    descriptions = df_subset['description'].values
    pdb_ids = df_subset['pdb_id'].values

    embeddings = {}
    for i, (desc, pdb_id) in tqdm(enumerate(zip(descriptions, pdb_ids)), desc="Text Embeddings", total=len(pdb_ids)):
        if pd.isna(desc):
            continue

        with torch.no_grad():
            try:
                tokenized = tokenizer(
                    desc,
                    truncation=True,
                    max_length=512,
                    padding="max_length",
                    return_tensors="pt"
                ).to(device)

                output = model(**tokenized)

                if args.embedding_type == "pooler":
                    des_embeddings = output.pooler_output
                else:  # "sequence"
                    des_embeddings = output.last_hidden_state

                embeddings[pdb_id] = des_embeddings[0].cpu()

                del tokenized, output, des_embeddings
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error processing text for PDB ID '{pdb_id}': {e}")
                continue

    output_file = args.output_dir / f"{args.output_prefix}_description.pkl"
    with open(output_file, "wb") as f:
        pickle.dump(embeddings, f)
    print(f"Text embeddings saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for antibody-antigen datasets.")

    # --- I/O Arguments ---
    parser.add_argument("--input_csv", type=Path, required=True,
                        help="Path to the input CSV file (e.g., train.csv or test.csv)")
    parser.add_argument("--output_dir", type=Path, required=True,
                        help="Directory to save the output .pkl embedding files.")
    parser.add_argument("--output_prefix", type=str, default="embeddings",
                        help="Prefix for output files (e.g., 'train', 'test')")

    # --- Model Cache Arguments ---
    parser.add_argument("--esmc_cache", type=Path, required=True,
                        help="Path to the .pth file for the ESMC model.")
    parser.add_argument("--scibert_cache", type=Path, required=True,
                        help="Path to the cache directory for SciBert (allenai/scibert_scivocab_uncased).")
    parser.add_argument("--igbert_cache", type=Path, required=True,
                        help="Path to the cache directory for IgBert (Exscientia/IgBert).")

    # --- Configuration Arguments ---
    parser.add_argument("--modality", type=str, choices=["antigen", "antibody", "text", "all"],
                        default="all",
                        help="Which embeddings to generate.")
    parser.add_argument("--embedding_type", type=str, choices=["pooler", "sequence"],
                        default="pooler",
                        help="Type of embedding to extract ('pooler' for mean-pooled, 'sequence' for all tokens).")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Torch device to use (e.g., 'cuda:0', 'cpu').")

    args = parser.parse_args()

    # --- Setup ---
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        df = pd.read_csv(args.input_csv)
    except FileNotFoundError:
        print(f"Error: Input CSV file not found at {args.input_csv}")
        return

    print(f"Processing file: {args.input_csv}")
    print(f"Output directory: {args.output_dir}")
    print(f"Models paths: ESMC='{args.esmc_cache}', SciBert='{args.scibert_cache}', IgBert='{args.igbert_cache}'")
    print(f"Config: Modality='{args.modality}', Embedding Type='{args.embedding_type}', Device='{args.device}'")

    # --- Run Embedding Generation ---
    if args.modality in ["all", "antigen"]:
        get_antigen_emb(df, args)

    if args.modality in ["all", "antibody"]:
        get_ab_emb(df, args)

    if args.modality in ["all", "text"]:
        get_text_emb(df, args)

    print("Embedding generation complete.")


if __name__ == "__main__":
    main()
