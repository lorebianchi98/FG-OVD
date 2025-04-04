import os
import subprocess
import argparse

def run_inference(dataset_path, model, tokenizer, output_base, n_hardnegatives_range):
    # Ensure the base output directory exists
    output_base = os.path.join(output_base, model.split('/')[-1])
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
            output_dir = os.path.join(output_base, f"{dataset}/{n_hardnegatives}.pkl")
            if os.path.isfile(output_dir):
                continue
            
            if dataset == 'transparency' and n_hardnegatives > 2:
                continue
            
            if dataset == 'pattern' and n_hardnegatives > 7:
                continue

            command = [
                "python", "owl_inference.py",
                "--dataset", benchmark_path,
                "--model", model,
                "--tokenizer", tokenizer,
                "--n_hardnegatives", str(n_hardnegatives),
                "--out", output_dir
            ]
            print(f"Running: {' '.join(command)}")

            # Run the command and capture the output
            try:
                subprocess.run(command, check=True)
                print(f"Completed for n_hardnegatives={n_hardnegatives}")
            except subprocess.CalledProcessError as e:
                print(f"Error running command for n_hardnegatives={n_hardnegatives}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Run owl_inference.py for a range of n_hardnegatives values.")
    parser.add_argument("--dataset_path", default='../../datasets/FG-OVD/benchmarks/', help="Path to the dataset JSON file.")
    parser.add_argument("--model", required=True, help="Path to the model weights.")
    parser.add_argument("--tokenizer", required=True, help="Tokenizer to use.")
    parser.add_argument("--output_base", default='predictions_fgovd', help="Base directory for output results.")
    parser.add_argument("--n_hardnegatives_start", type=int, default=0, help="Start value for n_hardnegatives (inclusive).")
    parser.add_argument("--n_hardnegatives_end", type=int, default=10, help="End value for n_hardnegatives (inclusive).")

    args = parser.parse_args()

    # Generate range for n_hardnegatives
    n_hardnegatives_range = range(args.n_hardnegatives_start, args.n_hardnegatives_end + 1)

    # Run inference for the specified range
    run_inference(
        dataset_path=args.dataset_path,
        model=args.model,
        tokenizer=args.tokenizer,
        output_base=args.output_base,
        n_hardnegatives_range=n_hardnegatives_range
    )

if __name__ == "__main__":
    main()
