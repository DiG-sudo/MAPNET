import argparse

from data import load_config, prepare_protein_data, project_path
from training import cleanup, resolve_device, select_best, train_one, write_csv


def main():
    parser = argparse.ArgumentParser(description="Search the DAMF branch count independently for each RBP.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--num-experts", nargs="+", type=int)
    parser.add_argument("--proteins", nargs="+", help="Optional subset for a partial run.")
    args = parser.parse_args()
    config = load_config(args.config)
    device = resolve_device(config)
    expert_grid = args.num_experts or config["model"]["num_experts_grid"]
    proteins = args.proteins or config["proteins"]
    output_dir = project_path(config, config["paths"]["output_dir"])
    all_results_path = output_dir / "grid_search_results.csv"
    best_results_path = output_dir / "grid_best_results.csv"

    for protein in proteins:
        splits, data_stats = prepare_protein_data(config, protein, device)
        candidate_rows = []
        for num_experts in expert_grid:
            print(f"[search] protein={protein} num_experts={num_experts}")
            row, state, model_config = train_one(
                config,
                protein,
                num_experts,
                splits,
                data_stats,
                device,
                evaluate_test=False,
            )
            candidate_rows.append(row)
            write_csv([row], all_results_path, replace_keys=("protein", "num_experts"))
            cleanup(state, model_config)
        best = select_best(candidate_rows)
        write_csv([best], best_results_path, replace_keys=("protein",))
        print(
            f"[best] protein={protein} num_experts={best['num_experts']} "
            f"val_auc={float(best['val_auc']):.6f}"
        )
        cleanup(splits, candidate_rows)


if __name__ == "__main__":
    main()
