CUDA_VISIBLE_DEVICES=3 python3 modelccu.py --device=cuda --batch-size=4 --lr=0.1 --include-utterance 
CUDA_VISIBLE_DEVICES=3 python3 modelccu.py --device=cuda --batch-size=4 --lr=0.01 --include-utterance 
CUDA_VISIBLE_DEVICES=3 python3 modelccu.py --device=cuda --batch-size=4 --condfident-only --include-utterance
CUDA_VISIBLE_DEVICES=3 python3 modelccu.py --device=cuda --batch-size=4 --regularisation=l1 --include-utterance
CUDA_VISIBLE_DEVICES=3 python3 modelccu.py --device=cuda --batch-size=4 --regularisation=dropout --include-utterance
CUDA_VISIBLE_DEVICES=3 python3 modelccu.py --device=cuda --batch-size=4 --lr=0.001 --include-utterance 
