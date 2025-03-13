# Think Low


## Architecture


![Architecture](/images/architecture.png)


## How to use


To generate the training dataset, which has thinking length greater than 100 tokens, you can run the following command:

```bash
python src/generate_solutions.py --output_filename bigmath_rl_solutions.jsonl --num_samples 1000
```

To generate the summary of the thinking, you can run the following command:

```bash
python src/generate_summary_thinking.py --input_file bigmath_rl_solutions.jsonl --output_dir data/generated
```

To generate the solutions for the test set, you can run the following command:

```bash
python src/generate_solutions.py --output_filename bigmath_rl_solutions_eval.jsonl --num_samples 1000 --eval_mode
```


To evaluate the performance of the model, you can run the following command:
