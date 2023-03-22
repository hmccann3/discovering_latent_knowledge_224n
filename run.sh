for NUM in 0 1 2 3 4 5 6 7 8 9 10 11 12
do 

	#python3 evaluate.py --cache_dir ../cache/ --model_name deberta-mnli --num_examples 1000 --batch_size 10 --dataset_name imdb --eval_dataset_name boolq --calibration_type sigmoid --confidence_loss_scale 1.0 --calibration_dataset_name imdb --split test --prompt_idx $NUM --eval_prompt_idx 4
	#python3 evaluate.py --cache_dir ../cache/ --model_name deberta-mnli --num_examples 1000 --batch_size 10 --dataset_name imdb --eval_dataset_name boolq --calibration_type sigmoid --confidence_loss_scale 1.0 --calibration_dataset_name imdb --split test --prompt_idx $NUM --eval_prompt_idx 4

	python3 evaluate.py --cache_dir ../cache/ --model_name deberta-mnli --num_examples 1000 --batch_size 10 --dataset_name boolq --prompt_idx $NUM --calibration_type dropout_loss --use_dropout_loss True --confidence_loss_scale 0.0 --dropout_factor 0.2
	python3 evaluate.py --cache_dir ../cache/ --model_name deberta-mnli --num_examples 1000 --batch_size 10 --dataset_name boolq --prompt_idx $NUM --calibration_type dropout --use_dropout True --confidence_loss_scale 0.0 --dropout_factor 0.2
	python3 evaluate.py --cache_dir ../cache/ --model_name deberta-mnli --num_examples 1000 --batch_size 10 --dataset_name boolq --prompt_idx $NUM --calibration_type dropout_and_confidence --use_dropout True --confidence_loss_scale 1.0 --dropout_factor 0.2
	python3 evaluate.py --cache_dir ../cache/ --model_name deberta-mnli --num_examples 1000 --batch_size 10 --dataset_name boolq --prompt_idx $NUM --calibration_type dropout_loss_and_confidence --use_dropout_loss True --confidence_loss_scale 1.0 --dropout_factor 0.2
	python3 evaluate.py --cache_dir ../cache/ --model_name deberta-mnli --num_examples 1000 --batch_size 10 --dataset_name boolq --prompt_idx $NUM --calibration_type sigmoid --confidence_loss_scale 1.0
	python3 evaluate.py --cache_dir ../cache/ --model_name deberta-mnli --num_examples 1000 --batch_size 10 --dataset_name boolq --prompt_idx $NUM --calibration_type isotonic --confidence_loss_scale 1.0

	#python3 evaluate.py --cache_dir ../cache/ --model_name deberta-mnli --num_examples 1000 --batch_size 10 --dataset_name imdb --prompt_idx $NUM --calibration_type dropout_and_loss --use_dropout_loss True --use_dropout True --confidence_loss_scale 0.0 --dropout_factor 0.2
	#python3 evaluate.py --cache_dir ../cache/ --model_name deberta-mnli --num_examples 1000 --batch_size 10 --dataset_name imdb --prompt_idx $NUM --calibration_type dropout_and_both_loss --use_dropout_loss True --use_dropout True --confidence_loss_scale 0.0 --dropout_factor 0.2


	#python3 evaluate.py --cache_dir ../cache/ --model_name deberta-mnli --num_examples 1000 --batch_size 10 --dataset_name imdb --calibration_dataset_name imdb --prompt_idx $NUM --calibration_type dropout --use_dropout True
	#python3 evaluate.py --cache_dir ../cache/ --model_name deberta-mnli --num_examples 1000 --batch_size 10 --dataset_name imdb --calibration_dataset_name imdb --prompt_idx $NUM --calibration_type dropout_loss --use_dropout_loss True
	#python3 evaluate.py --cache_dir ../cache/ --model_name deberta-mnli --num_examples 1000 --batch_size 10 --dataset_name imdb --calibration_dataset_name imdb --prompt_idx $NUM --calibration_type dropout_and_loss --use_dropout True --use_dropout_loss True
	#python3 evaluate.py --cache_dir ../cache/ --model_name deberta-mnli --num_examples 1000 --batch_size 10 --dataset_name imdb --calibration_dataset_name imdb --prompt_idx $NUM --calibration_type sigmoid
	#python3 evaluate.py --cache_dir ../cache/ --model_name deberta-mnli --num_examples 1000 --batch_size 10 --dataset_name imdb --calibration_dataset_name imdb --prompt_idx $NUM --calibration_type isotonic
done
