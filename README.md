
mnist
python --data_name=Mnist --batch_size=1500 

python main_pl_li.py  --data_name=Coil20 --batch_size=1440 --vs=1e-3 --perplexity=20 --epochs=1000 --near_bound=0.5 --far_bound=1

python main_pl_li.py --data_name=SwissRoll --perplexity=20 --vs=1e-4 --epochs=3000