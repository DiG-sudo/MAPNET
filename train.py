import argparse

from data import load_config, prepare_protein_data, project_path
from training import cleanup, read_csv, resolve_device, save_checkpoint, train_one, write_csv


def main():
    parser = argparse.ArgumentParser(description="Retrain MAPNet with the validation-selected branch count.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--use-search-results",
        default="outputs/grid_best_results.csv",
        help="CSV produced by search.py.",
    )
    parser.add_argument("--proteins", nargs="+", help="Optional subset for a partial run.")
    args = parser.parse_args()
    config = load_config(args.config)
    device = resolve_device(config)
    search_path = project_path(config, args.use_search_results)
    search_rows = read_csv(search_path)
    selected = {row["protein"]: int(row["num_experts"]) for row in search_rows}
    proteins = args.proteins or config["proteins"]
    missing = [protein for protein in proteins if protein not in selected]
    if missing:
        raise KeyError(f"Missing grid-search results for: {', '.join(missing)}")
    output_dir = project_path(config, config["paths"]["output_dir"])
    result_path = output_dir / "test_results.csv"

    for protein in proteins:
        num_experts = selected[protein]
        print(f"[train] protein={protein} num_experts={num_experts}")
        splits, data_stats = prepare_protein_data(config, protein, device)
        row, state, model_config = train_one(
            config,
            protein,
            num_experts,
            splits,
            data_stats,
            device,
            evaluate_test=True,
        )
        checkpoint = save_checkpoint(config, protein, model_config, state, row)
        write_csv([row], result_path, replace_keys=("protein",))
        print(
            f"[test] protein={protein} auc={float(row['test_auc']):.6f} "
            f"f1={float(row['test_f1']):.6f} checkpoint={checkpoint}"
        )
        cleanup(splits, state, model_config)


if __name__ == "__main__":
    main()
