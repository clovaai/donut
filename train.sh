DS=/home/abaybektursun/chart-gen/generated_stacked_bar/jsonl
/app/donut/venv/bin/python train.py --config config/train_cord.yaml \
                --dataset_name_or_paths '["./jsonl"]' \
                --exp_version "abay_donut_28"    
