import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from copy import deepcopy
import gc
import torch
import itertools
import hydra
from lightning import seed_everything
from omegaconf import DictConfig
import pandas as pd
import csv
import rootutils
import time
from typing import List
from src.models.modules.predictor import ContrastiveModel
from src.models.modules.predictor import CrossAttentionModel
from src.models.modules.seqEncoder import IgBertEncoder
from src.data.training_data_module import TrainingDataModule, CustomCollateFn
from src.models.CVAE import CVAE

root_dir = os.path.abspath(__file__)
root = rootutils.setup_root(root_dir, pythonpath=True)


def load_config(config_path: str):
    with open(config_path, "r") as f:
        raw_cfg = yaml.safe_load(f)
        cfg = OmegaConf.create(raw_cfg)
        return cfg


def log_optimization_result(log_path, index, pdb_id, antigen, original_ab, optimized_ab, original_fitness,
                            original_predictor_fitness, optimized_fitness):
    """
    Logs optimization results to a CSV file.

    Parameters:
        log_path (str): Path to the CSV file.
        index (int): Index of the sample.
        antigen (str): Antigen sequence.
        original_ab (str): Original Antibody sequence.
        optimized_ab (str): Optimized Antibody sequence.
        original_fitness (float): Fitness of the original sequence.
        optimized_fitness (float): Fitness of the optimized sequence.
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    log_entry = {
        "index": index,
        "pdb_id": pdb_id,
        "antigen": antigen,
        "original_ab": original_ab,
        "optimized_ab": optimized_ab,
        # "original_fitness": original_fitness,
        "original_predictor_fitness": original_predictor_fitness,
        "optimized_fitness": optimized_fitness
    }

    # Append or create
    if os.path.exists(log_path):
        log_df = pd.read_csv(log_path)
        log_df = pd.concat([log_df, pd.DataFrame([log_entry])], ignore_index=True)
    else:
        log_df = pd.DataFrame([log_entry])

    log_df.to_csv(log_path, index=False)


def log_elite_items(log_file_path, generation_index, elite_items):
    """
    Logs the elite sequences and their fitness scores to a file.

    Parameters:
    - log_file_path (str): Path to the log file.
    - generation_index (int): Current generation number.
    - elite_items (List[Tuple[str, float]]): List of (sequence, fitness) tuples.
    """
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    file_exists = os.path.isfile(log_file_path)

    with open(log_file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["Generation", "Sequence", "Fitness"])

        for seq, fitness in elite_items:
            writer.writerow([generation_index, seq, fitness])


def calc_fitness(opt_ab_seqs: List, ag_emb, ab_encoder: IgBertEncoder, aff_predictor):
    with torch.inference_mode():
        # embed the new antibody and get the embedding of the antigen
        batch_ab_emb = ab_encoder.forward(opt_ab_seqs)

        # predict the affinity with cross_attention_model
        ag_emb_expanded = ag_emb.expand(batch_ab_emb.shape[0], -1)
        affinities = aff_predictor(batch_ab_emb.cuda(), ag_emb_expanded.cuda())
        del batch_ab_emb, ag_emb
        return affinities


def generate_valid_antibody(model, y_ab, ag_emb, des_emb, max_retries=10):
    for _ in range(max_retries):
        error = False
        new_seqs = model.reconstruct_from_wt(y_ab, ag_emb, des_emb)
        for generated_seq in new_seqs:
            if not isinstance(generated_seq, str):
                error = True
                break

            # Check format: must contain exactly one '|'
            if "|" in generated_seq:
                parts = generated_seq.split("|")
                if len(parts) != 2:
                    error = True
        if not error:
            return new_seqs
        else:
            print("Failed to generate valid antibody sequence.")
            print("Generated antibody sequence:", new_seqs)
            continue

    return []


# MARK: PERFORM DIRECTED EVOLUTION
def perform_directed_evolution(output_folder, opt_ab_seqs, ag_emb, des_emb,
                               model: CVAE, ab_encoder, aff_predictor, cfg):
    batch_size = cfg.recon_wt_batch
    init_scores = calc_fitness(opt_ab_seqs, ag_emb, ab_encoder, aff_predictor)

    # sort the initial sequences by their scores (ascending: lower ΔG = better)
    items = list(zip(opt_ab_seqs, init_scores))
    items.sort(key=lambda x: x[1])

    items = items[:cfg.num_elites]
    cur_items = items

    for i in range(cfg.num_generations):
        # duplicate the current items for beam search
        cur_items = list(itertools.chain.from_iterable(
            list(deepcopy(it) for _ in range(cfg.beam_size))
            for it in cur_items
        ))
        cur_seqs = list(map(lambda x: x[0], cur_items))

        # chunk the current sequences into batches
        cur_seq_chunks = [cur_seqs[i:i + batch_size] for i in range(0, len(cur_seqs), batch_size)]
        all_new_seqs = []

        for cur_seq_batch in cur_seq_chunks:
            y_ab = ab_encoder(cur_seq_batch)
            # perform directed evolution by "reconstructing" sequences
            new_seqs = generate_valid_antibody(model, y_ab, ag_emb, des_emb)
            del y_ab
            if len(new_seqs) != 0:
                all_new_seqs.extend(new_seqs)

        # calculate the fitness of the new sequences
        new_scores = calc_fitness(all_new_seqs, ag_emb, ab_encoder, aff_predictor)
        new_items = list(zip(all_new_seqs, new_scores))

        # combine the current items with the new items and sort them, get the top elite sequences
        items = cur_items + new_items
        items.sort(key=lambda x: x[1])
        items = items[:cfg.num_elites]

        cur_items = items
        log_file_path = os.path.join(output_folder, "generation_details.csv")
        log_elite_items(log_file_path, i, cur_items)
        # print(f"Generation {i} done with result: {new_items}")

    return cur_items


# MARK: MAIN OPTIMIZATION FUNCTION
@hydra.main(config_path='../configs', config_name='optimize')
def optimize(cfg: DictConfig) -> None:
    seed_everything(cfg.seed, workers=True)

    cvae, aff_predictor, ab_encoder = load_and_initialize_models(cfg)

    # Build data module and use the test split for optimization.
    data_module = TrainingDataModule(cfg)
    data_module.setup()
    full_loader = data_module.test_dataloader()

    # directed evolution
    for i, data in enumerate(full_loader):
        gc.collect()
        torch.cuda.empty_cache()
        pdb_id, ab_seq, ag_seq, ab_emb, ag_emb, des_emb, orig_fitness = extract_info(data)

        # denormalize the delta_g value
        orig_fitness = orig_fitness  # * (delta_g_max - delta_g_min) + delta_g_min

        wt_seq = ab_seq
        opt_seqs = [wt_seq] * cfg.num_samples

        output_folder = str(pdb_id) + "_" + str(i)
        output_folder = f"{cfg.log_folder}/{output_folder}"
        items = perform_directed_evolution(output_folder, opt_seqs, ag_emb, des_emb,
                                           cvae, ab_encoder, aff_predictor, cfg)

        original_predictor_fitness = calculate_original_predictor_fitness(aff_predictor, ab_emb, ag_emb)

        # get the best optimized sequence
        best_fitness = original_predictor_fitness
        opt_seq = ""
        for (opt_ab_seq, fitness) in items:
            if fitness < best_fitness:
                best_fitness = fitness
                opt_seq = opt_ab_seq

        print(
            f"Original seq: {ab_seq} - Best seq: {opt_ab_seq} - Original fitness: {original_predictor_fitness} - Best fitness: {best_fitness}")

        log_path = f"{cfg.log_folder}/optimization_log.csv"
        log_optimization_result(
            log_path,
            index=i,
            pdb_id=pdb_id,
            antigen=ag_seq,
            original_ab=ab_seq,
            optimized_ab=opt_seq,
            original_fitness=float(orig_fitness),
            original_predictor_fitness=original_predictor_fitness,
            optimized_fitness=float(best_fitness)
        )

        del ab_emb, ag_emb
        gc.collect()
        torch.cuda.empty_cache()


def calculate_original_predictor_fitness(aff_predictor, ab_emb, ag_emb):
    original_predictor_fitness = aff_predictor(ab_emb.cuda(), ag_emb.cuda())
    original_predictor_fitness = original_predictor_fitness[0].item()
    return original_predictor_fitness


def extract_info(data):
    pdb_id = data['pdb_id'][0]
    ab_seq = data["ab_sequences"][0]
    ag_seq = data["ag_sequences"][0]
    ab_emb = data["ab_embedding"].to("cuda")
    ag_emb = data["ag_embedding"].to("cuda")
    des_emb = data["des_embedding"].to("cuda")
    return pdb_id, ab_seq, ag_seq, ab_emb, ag_emb, des_emb, 0.0


def load_and_initialize_models(cfg):
    # Auto-select checkpoint based on use_description when model_path is not explicitly set.
    explicit_path = cfg.get("model_path", None)
    if explicit_path:
        ckpt_path = explicit_path
    elif cfg.model.get("use_description", 1):
        ckpt_path = cfg.get("model_path_w_des", "checkpoints/cvae/best_w_des.ckpt")
    else:
        ckpt_path = cfg.get("model_path_wo_des", "checkpoints/cvae/best_wo_des.ckpt")

    cvae = CVAE.load_from_checkpoint(os.path.join(root, ckpt_path), cfg=cfg)
    cvae.eval()

    proj_model = ContrastiveModel(ab_embed_dim=cfg.model.ab_dimen, ag_embed_dim=cfg.model.ag_dimen,
                                  embed_dim=cfg.model.predictor.pred_hidden_dim)
    proj_model.load_state_dict(torch.load(os.path.join(root,cfg.model.predictor.contrastive_model_path)))
    proj_model.eval()
    cross_attention_model = CrossAttentionModel(proj_model, proj_embed_dim=cfg.model.predictor.pred_hidden_dim,
                                                attn_embed_dim=cfg.model.predictor.pred_hidden_dim)
    cross_attention_model.load_state_dict(torch.load(os.path.join(root,cfg.model.predictor.cross_attention_model_path)))
    cross_attention_model.eval()
    aff_predictor = cross_attention_model
    aff_predictor = aff_predictor.cuda()

    ab_encoder = IgBertEncoder(cfg)
    ab_encoder.to("cuda:0").eval()
    return cvae, aff_predictor, ab_encoder


if __name__ == '__main__':
    # start timer
    timeStart = time.time()
    optimize()
    # end timer
    timeEnd = time.time()
    print("Time taken: ", timeEnd - timeStart)
