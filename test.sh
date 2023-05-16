CUDA_VISIBLE_DEVICES=2 python3 modelccu.py --device=cuda --batch-size=4 
CUDA_VISIBLE_DEVICES=2 python3 modelccu.py --device=cuda --batch-size=4 --lr=0.0001 --include-utterance
CUDA_VISIBLE_DEVICES=2 python3 modelccu.py --device=cuda --batch-size=4 --downsample=4 --include-utterance
CUDA_VISIBLE_DEVICES=2 python3 modelccu.py --device=cuda --batch-size=4 --lrscheduler --include-utterance
CUDA_VISIBLE_DEVICES=2 python3 modelccu.py --device=cuda --batch-size=4 --classifierlayers=2 --include-utterance


