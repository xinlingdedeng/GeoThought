python3 inference.py \
  --api_url "http://127.0.0.1:8111" \
  --model_name "model_name" \
  --prompt_path "XXX/geoqa_test_prompts.jsonl" \
  --image_root "XXX/prompts/" \
  --output_path "./result_geoqa/results_model_name.jsonl" \
  --max_workers 5
