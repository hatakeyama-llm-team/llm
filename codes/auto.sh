cd 1_load_dataset/
python integrate_dataset.py 
cd ../2_pretrain
python 1_train_sentencepiece_tokenizer.py 
bash 2_train_node1.sh 