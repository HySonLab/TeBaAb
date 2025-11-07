import random
import math
import itertools
from typing import List, Tuple, Dict
from sklearn.feature_extraction.text import TfidfVectorizer


class AntibodyRandomMasker:
    """Random masking strategy for antibody sequences"""

    def __init__(self, k: int = 1, max_subs: int = 5, mask_token: str = "*",
                 mask_heavy: bool = True, mask_light: bool = True):
        """
        Args:
            k: Length of k-mer to mask
            max_subs: Maximum number of substitutions per chain
            mask_token: Token to use for masking
            mask_heavy: Whether to mask heavy chain
            mask_light: Whether to mask light chain
        """
        self.k = k
        self.mask_token = mask_token
        self.max_subs = max_subs
        self.mask_heavy = mask_heavy
        self.mask_light = mask_light

    def mask_chain(self, chain: str) -> Tuple[str, List[int]]:
        """Mask random positions in a single chain"""
        # if self.k > 1:
        #     assert self.max_subs == 1, "Only substitute 1 k-mer at a time for k > 1."

        lseq = list(chain)
        min_pos = 0
        max_pos = len(lseq) - self.k + 1

        if max_pos <= min_pos:
            return chain, []

        candidate_masked_pos = list(range(min_pos, max_pos))
        random.shuffle(candidate_masked_pos)
        pos_to_mutate = candidate_masked_pos[:min(self.max_subs, len(candidate_masked_pos))]

        for pos in pos_to_mutate:
            lseq[pos:pos + self.k] = [self.mask_token] * self.k

        if self.k == 1:
            return ''.join(lseq), pos_to_mutate
        else:
            # For k > 1, return all positions in the k-mer
            return ''.join(lseq), list(range(pos_to_mutate[0], pos_to_mutate[0] + self.k))

    def run(self, population: List[str]) -> Tuple[List[str], List[Dict[str, List[int]]]]:
        """
        Mask antibody sequences in population

        Args:
            population: List of antibody sequences in format "heavy|light"

        Returns:
            masked_population: List of masked sequences
            masked_positions: List of dicts with 'heavy' and 'light' keys containing masked positions
        """
        masked_population = []
        masked_positions = []

        for seq in population:
            # Split into heavy and light chains
            parts = seq.split("|")
            if len(parts) != 2:
                raise ValueError(f"Invalid antibody format: {seq}")

            heavy_chain, light_chain = parts[0].strip(), parts[1].strip()
            positions_dict = {'heavy_chain': [], 'light_chain': []}

            # Mask heavy chain
            if self.mask_heavy:
                masked_heavy, heavy_pos = self.mask_chain(heavy_chain)
                positions_dict['heavy_chain'] = heavy_pos
            else:
                masked_heavy = heavy_chain

            # Mask light chain
            if self.mask_light:
                masked_light, light_pos = self.mask_chain(light_chain)
                positions_dict['light_chain'] = light_pos
            else:
                masked_light = light_chain

            # Combine back
            masked_seq = f"{masked_heavy}|{masked_light}"
            masked_population.append(masked_seq)
            masked_positions.append(positions_dict)

        return masked_population, masked_positions


class AntibodyImportanceMasker:
    """Importance-based masking strategy for antibody sequences"""

    def __init__(self, k: int = 3, max_subs: int = 5, mask_token: str = "*",
                 low_importance_mask: bool = True, mask_heavy: bool = True,
                 mask_light: bool = True):
        """
        Args:
            k: Length of k-mer to mask
            max_subs: Maximum number of substitutions per chain
            mask_token: Token to use for masking
            low_importance_mask: If True, mask low importance regions; else mask high importance
            mask_heavy: Whether to mask heavy chain
            mask_light: Whether to mask light chain
        """
        self.k = k
        self.max_subs = max_subs
        self.mask_token = mask_token
        self.low_importance_mask = low_importance_mask
        self.mask_heavy = mask_heavy
        self.mask_light = mask_light
        self.tfidf = TfidfVectorizer(lowercase=False, token_pattern=r"(?u)\b\w+\b")

    def split_kmers(self, seqs: List[str]) -> List[List[str]]:
        """Split sequences into k-mers"""
        return [[seq[i:i + self.k] for i in range(len(seq) - self.k + 1)] for seq in seqs]

    def _get_entropy_of_unique_tokens(self, seqs: List[List[str]]) -> Dict[str, float]:
        """Calculate entropy for each unique k-mer"""
        bag_of_toks = list(itertools.chain.from_iterable(seqs))
        set_toks = set(bag_of_toks)
        count = {tok: bag_of_toks.count(tok) for tok in set_toks}

        entropy = {}
        for k, v in count.items():
            prob = v / len(bag_of_toks)
            entropy[k] = -1.0 * prob * math.log(prob) if prob > 0 else 0

        return entropy

    def _measure_importance(self, sequences: List[List[str]]) -> List[Dict[str, float]]:
        """Measure importance of k-mers using TF-IDF and entropy"""
        if not sequences or not any(sequences):
            return []

        # Filter out empty sequences
        valid_seqs = [seq for seq in sequences if seq]
        if not valid_seqs:
            return []

        merge_seqs = [' '.join(seq) for seq in valid_seqs]

        # Run TF-IDF
        tfidfs = self.tfidf.fit_transform(merge_seqs)
        actual_vocabs = {
            name: idx for idx, name in enumerate(self.tfidf.get_feature_names_out())
        }

        # Get entropy
        kmer2entropy = self._get_entropy_of_unique_tokens(valid_seqs)

        # Measure importance
        importances = []
        for seq_idx, seq in enumerate(valid_seqs):
            kmer2imp = dict()
            setseq = list(set(seq))
            seq_tfidf = tfidfs[seq_idx].sum()
            seq_entropy = sum(kmer2entropy.get(kmer, 0) for kmer in setseq)

            for kmer in setseq:
                if kmer in actual_vocabs:
                    kmer_idx = actual_vocabs[kmer]
                    tfidf = tfidfs[seq_idx, kmer_idx]

                    if seq_tfidf > 0 and seq_entropy > 0:
                        kmer2imp[kmer] = tfidf / seq_tfidf + kmer2entropy.get(kmer, 0) / seq_entropy
                    elif seq_tfidf > 0:
                        kmer2imp[kmer] = tfidf / seq_tfidf
                    else:
                        kmer2imp[kmer] = 0.5  # Default importance

            importances.append(kmer2imp)

        return importances

    def mask_chain_by_importance(self, chain: str, kmer_seq: List[str],
                                 kmer2imp: Dict[str, float]) -> Tuple[str, List[int]]:
        """Mask chain based on k-mer importance"""
        if not kmer2imp:
            return chain, []

        # if self.k > 1:
        #     assert self.max_subs == 1, "Only substitute 1 k-mer at a time for k > 1."

        # Sort k-mers by importance
        if self.low_importance_mask:
            sorted_kmers = sorted(kmer2imp.items(), key=lambda x: x[1])
        else:
            sorted_kmers = sorted(kmer2imp.items(), key=lambda x: x[1], reverse=True)

        positions = []
        lseq = list(chain)
        masked_count = 0

        for kmer, _ in sorted_kmers:
            if masked_count >= self.max_subs:
                break

            # Find positions of this k-mer in sequence
            for i in range(len(kmer_seq)):
                if kmer_seq[i] == kmer and i not in positions:
                    # Mask this position
                    lseq[i:i + self.k] = [self.mask_token] * self.k
                    positions.append(i)
                    masked_count += 1
                    break

        if self.k == 1:
            return ''.join(lseq), positions
        else:
            # For k > 1, return all positions in the k-mer
            if positions:
                return ''.join(lseq), list(range(positions[0], positions[0] + self.k))
            return chain, []

    def run(self, population: List[str]) -> Tuple[List[str], List[Dict[str, List[int]]]]:
        """
        Mask antibody sequences based on importance

        Args:
            population: List of antibody sequences in format "heavy|light"

        Returns:
            masked_population: List of masked sequences
            masked_positions: List of dicts with 'heavy' and 'light' keys containing masked positions
        """
        masked_population = []
        masked_positions = []

        # Separate heavy and light chains
        heavy_chains = []
        light_chains = []
        for seq in population:
            parts = seq.split("|")
            if len(parts) != 2:
                raise ValueError(f"Invalid antibody format: {seq}")
            heavy_chains.append(parts[0].strip())
            light_chains.append(parts[1].strip())

        # Calculate importance for heavy chains
        heavy_importances = []
        if self.mask_heavy and heavy_chains:
            heavy_kmers = self.split_kmers(heavy_chains)
            heavy_importances = self._measure_importance(heavy_kmers)

        # Calculate importance for light chains
        light_importances = []
        if self.mask_light and light_chains:
            light_kmers = self.split_kmers(light_chains)
            light_importances = self._measure_importance(light_kmers)

        # Mask each sequence
        for i, (heavy, light) in enumerate(zip(heavy_chains, light_chains)):
            positions_dict = {'heavy_chain': [], 'light_chain': []}

            # Mask heavy chain
            if self.mask_heavy and i < len(heavy_importances):
                heavy_kmer_seq = [heavy[j:j + self.k] for j in range(len(heavy) - self.k + 1)]
                masked_heavy, heavy_pos = self.mask_chain_by_importance(
                    heavy, heavy_kmer_seq, heavy_importances[i]
                )
                positions_dict['heavy_chain'] = heavy_pos
            else:
                masked_heavy = heavy

            # Mask light chain
            if self.mask_light and i < len(light_importances):
                light_kmer_seq = [light[j:j + self.k] for j in range(len(light) - self.k + 1)]
                masked_light, light_pos = self.mask_chain_by_importance(
                    light, light_kmer_seq, light_importances[i]
                )
                positions_dict['light_chain'] = light_pos
            else:
                masked_light = light

            # Combine back
            masked_seq = f"{masked_heavy}|{masked_light}"
            masked_population.append(masked_seq)
            masked_positions.append(positions_dict)

        return masked_population, masked_positions
