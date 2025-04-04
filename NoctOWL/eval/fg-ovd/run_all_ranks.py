import os
import subprocess
import argparse

def run_inference(dataset_path, model_predictions, output_base, n_hardnegatives_range):
    # Ensure the base output directory exists
    output_base = os.path.join(output_base, model_predictions.split('/')[-1])
    os.makedirs(output_base, exist_ok=True)
    datasets = [
        'color',
        'transparency',
        'pattern',
        'material',
        'shuffle_negatives',
        '1_attributes',
        '2_attributes',
        '3_attributes',
    ]
    for dataset in datasets:
        benchmark_path = os.path.join(dataset_path, dataset + '.json')
        for n_hardnegatives in n_hardnegatives_range:
            output_dir = os.path.join(output_base, f"{dataset}/{n_hardnegatives}.json")
            predictions_path = os.path.join(model_predictions, f"{dataset}/{n_hardnegatives}.pkl")
            if os.path.isfile(output_dir):
                continue
            
            if dataset == 'transparency' and n_hardnegatives > 2:
                continue
            
            if dataset == 'pattern' and n_hardnegatives > 7:
                continue
            # python evaluate_map.py --ground_truth benchmarks/1_attributes.json --predictions results/base.pkl --out 
            command = [
                "python", "ranks.py",
                "--gt", benchmark_path,
                "--preds", predictions_path,
                "--out", output_dir,
                "--n_neg", str(n_hardnegatives)
            ]
            print(f"Running: {' '.join(command)}")

            # Run the command and capture the output
            try:
                subprocess.run(command, check=True)
                print(f"Completed for n_hardnegatives={n_hardnegatives}")
            except subprocess.CalledProcessError as e:
                print(f"Error running command for n_hardnegatives={n_hardnegatives}: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default='../../datasets/FG-OVD/benchmarks/', help="Path to the dataset JSON file.")
    parser.add_argument("--model_predictions", required=True, help="Path to the model predictions.")
    parser.add_argument("--output_base", default='ranks', help="Base directory for output results.")
    parser.add_argument("--n_hardnegatives_start", type=int, default=0, help="Start value for n_hardnegatives (inclusive).")
    parser.add_argument("--n_hardnegatives_end", type=int, default=10, help="End value for n_hardnegatives (inclusive).")

    args = parser.parse_args()

    # Generate range for n_hardnegatives
    n_hardnegatives_range = range(args.n_hardnegatives_start, args.n_hardnegatives_end + 1)

    # Run inference for the specified range
    run_inference(
        dataset_path=args.dataset_path,
        model_predictions=args.model_predictions,
        output_base=args.output_base,
        n_hardnegatives_range=n_hardnegatives_range
    )

if __name__ == "__main__":
    main()
