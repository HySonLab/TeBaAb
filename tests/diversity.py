import csv
import sys
import Levenshtein


def calculate_diversity(sequences):
    N = len(sequences)
    if N < 2:
        return 0.0
    total_distance = 0
    for i in range(N - 1):
        for j in range(i + 1, N):
            dist = Levenshtein.distance(sequences[i], sequences[j])
            total_distance += dist
    return (2 / (N * (N - 1))) * total_distance


def calculate_novelty(generated_sequences, training_sequences):
    if not generated_sequences or not training_sequences:
        return 0.0
    min_distances = []
    for s in generated_sequences:
        min_dist = min(Levenshtein.distance(s, s_prime) for s_prime in training_sequences)
        min_distances.append(min_dist)
    return sum(min_distances) / len(min_distances)


def load_antibodies_from_csv(file_path, column_name):
    heavy_chains = []
    light_chains = []
    ab_chains = []
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            ab = row[column_name]
            try:
                heavy, light = ab.split('|')
                heavy_chains.append(heavy.strip())
                light_chains.append(light.strip())
                ab_chains.append(ab.strip())
            except ValueError:
                print(f"⚠️  Skipping malformed antibody sequence: {ab}")
                continue
    return heavy_chains, light_chains, ab_chains


def load_training_csv(file_path):
    heavy_chains = []
    light_chains = []
    ab_chains = []
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            heavy = row["heavy"]
            light = row["light"]
            ab = heavy + "|" + light
            heavy_chains.append(heavy.strip())
            light_chains.append(light.strip())
            ab_chains.append(ab.strip())
    return heavy_chains, light_chains, ab_chains


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 tests/diversity.py <path_to_csv_file>")
        sys.exit(1)
    csv_file = sys.argv[1]
    print("🔍 Loading and processing antibody sequences...")

    heavy_orig, light_orig, ab_orig = load_training_csv("datasets/abdes/train.csv")
    heavy_opt, light_opt, ab_opt = load_antibodies_from_csv(csv_file, 'optimized_ab')

    print("\n📊 Calculating diversity scores...")
    # diversity_heavy_orig = calculate_diversity(heavy_orig)
    # diversity_light_orig = calculate_diversity(light_orig)
    diversity_heavy_opt = calculate_diversity(heavy_opt)
    diversity_light_opt = calculate_diversity(light_opt)
    # diversity_ab_orig = calculate_diversity(ab_orig)
    diversity_ab_opt = calculate_diversity(ab_opt)
    #
    print("\n🧪 Diversity Results:")
    # print(f"Original Heavy Chain Diversity:   {diversity_heavy_orig:.4f}")
    # print(f"Original Light Chain Diversity:   {diversity_light_orig:.4f}")
    print(f"Optimized Heavy Chain Diversity:  {diversity_heavy_opt:.4f}")
    print(f"Optimized Light Chain Diversity:  {diversity_light_opt:.4f}")
    # print(f"Original AB Chain Diversity:      {diversity_ab_orig:.4f}")
    print(f"Optimized AB Chain Diversity:     {diversity_ab_opt:.4f}")

    print("\n✨ Calculating novelty scores...")
    novelty_heavy_opt = calculate_novelty(heavy_opt, heavy_orig)
    novelty_light_opt = calculate_novelty(light_opt, light_orig)
    novelty_ab_opt = calculate_novelty(ab_opt, ab_orig)

    print("\n🌟 Novelty Results (Optimized vs Original):")
    print(f"Optimized Heavy Chain Novelty:    {novelty_heavy_opt:.4f}")
    print(f"Optimized Light Chain Novelty:    {novelty_light_opt:.4f}")
    print(f"Optimized AB Chain Novelty:       {novelty_ab_opt:.4f}")


if __name__ == '__main__':
    main()
