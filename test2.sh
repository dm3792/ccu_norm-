CUDA_VISIBLE_DEVICES=3 python3 modelccu.py --device=cuda --batch-size=4 --regularisation=dropout --include-utterance
CUDA_VISIBLE_DEVICES=3 python3 modelccu.py --device=cuda --batch-size=4 --downsample=4 --include-utterance
CUDA_VISIBLE_DEVICES=3 python3 modelccu.py --device=cuda --batch-size=4 --lrscheduler --include-utterance
CUDA_VISIBLE_DEVICES=3 python3 modelccu.py --device=cuda --batch-size=4 --classifierlayers=2 --include-utterance
CUDA_VISIBLE_DEVICES=3 python3 modelccu.py --device=cuda --batch-size=4 --condfident-only --include-utterance
