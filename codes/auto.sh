cd 1_load_dataset/
python integrate_dataset.py 

cd ../2_pretrain
python 1_train_sentencepiece_tokenizer.py 
bash 2_tokenize.sh
bash 3_train_node1.sh 
bash 4_convert_to_HF.sh
python 6_upload.py