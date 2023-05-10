[
    'u':{" --include-utterance", ""},
    'r':{" --lr=l1",""," --lr=dropout"},
    'd':{" "," --downsample=4"},
    'lr':{" --lr=0.1"," --lr=0.001"," --lr=0.0001"," --lr=0.00001",""},
    "lrs":{""," --lrscheduler"},
    "cl":{" --classifierlayers=2",""},
    "cf"={""," --confident-only"}
]


"python3 modelccu.py --device=cuda --batch-size=4 --lr=0.1 --include-utterance "
"python3 modelccu.py --device=cuda --batch-size=4 --lr=0.01 --include-utterance "
"python3 modelccu.py --device=cuda --batch-size=4 --lr=0.001 --include-utterance "
"python3 modelccu.py --device=cuda --batch-size=4 --lr=0.0001 --include-utterance"
"python3 modelccu.py --device=cuda --batch-size=4 "
"python3 modelccu.py --device=cuda --batch-size=4 --regularisation=l1 --include-utterance"
"python3 modelccu.py --device=cuda --batch-size=4 --regularisation=dropout --include-utterance"
"python3 modelccu.py --device=cuda --batch-size=4 --downsample=4 --include-utterance"
"python3 modelccu.py --device=cuda --batch-size=4 --lrscheduler --include-utterance"
"python3 modelccu.py --device=cuda --batch-size=4 --classifierlayers=2 --include-utterance"
"python3 modelccu.py --device=cuda --batch-size=4 --condfident-only --include-utterance"




