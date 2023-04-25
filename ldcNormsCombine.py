import pandas as pd
import loaders 
from loaders import ldc_data

def generate_input(split,utt_before,utt_after):
    # with open('amith-cache.pkl', 'rb') as pickle_file:
    #     content = pickle.load(pickle_file)
    print(split)
    content = ldc_data.load_ldc_data()
    result =[]
    norm_val = {
    101:'apology',
    102:'criticism',
    103:'greeting',
    104:'request',
    105:'persuasion',
    106:'thanks',
    107:'leave',
    108 :'humour',
    109 :'embarassment',
    110 :'command',
    111 :'congrats',
    112 :'interest',
    113 :'concern',
    114 :'encouragement',
    115 :'empathy',
    116 :'feedback',
    117 :'trust',
    118 :'respect',
    119 :'flattery'
    }
    for item in content:
        
        
        if(content[item]['split']!=split):
            continue

        try:
            norms = pd.read_csv('yi/'+item+'.tab', sep='\t', lineterminator='\n')
            print('success')
        except:
            continue
        
        

        start = content[item]['start']
        end = content[item]['end']
        cpu = []
        for chp in content[item]['changepoints']:
            cpu.append((float(chp['timestamp']),'c'))
        
        for ut in content[item]['utterances']:
            cpu.append((float(ut['start']),'u',float(ut['end']),ut['text']))
            
        
        cpu.sort(key=lambda x: x[0])
        l = len(cpu)
        for idx,box in enumerate(cpu):
            if len(box)==4:
                cp = 0
                start_time=box[0]
                dt = content[item]['data_type']
                if box[0]>=start and box[2]<=end:
                    if idx-1>=0 and len(cpu[idx-1])==2:
                        if(cpu[idx-1][0]>=box[0] and cpu[idx-1][0]<=box[2]):
                            cp=1

                    if cp==0 and idx+1<l and len(cpu[idx+1])==2:
                        if(cpu[idx+1][0]>=box[0] and cpu[idx+1][0]<=box[2]):
                            cp=1

                    itr = idx
                    cnt=0
                    utt=box[3]
                    ustart = box[0]
                    uend = box[2]
                    while cnt<utt_before and itr>0:
                        if len(cpu[itr])==4:
                            cnt+=1
                            utt = cpu[itr][3] + utt
                            ustart = cpu[itr][0]
                        itr-=1

                    itr=idx
                    cnt=0    
                    while cnt<utt_after and itr<l:
                        if len(cpu[itr])==4:
                            cnt+=1
                            utt = cpu[itr][3] + utt
                            uend = cpu[itr][2]
                        itr+=1

                    df = norms.reset_index()  
                    a = df.index[(ustart<df['start']) & (uend>df['end'])].tolist()
                    b = df.index[(ustart>df['start']) & (ustart<df['end']) & (uend>df['end'])].tolist()
                    c = df.index[(ustart<df['start']) & (uend>df['start']) & (uend<df['end'])].tolist()
                    norm_list = a+b+c
                    ad = set()
                    vi = set()
                    for i in norm_list:
                        if df.loc[[i]]['status'].values[0]=='adhere':
                            ad.add(df.loc[[i]]['norm'].values[0])
                        if df.loc[[i]]['status'].values[0]=='violate':
                            ad.add(df.loc[[i]]['norm'].values[0])

                    norm_string = "ADHERE:"
                    for i in ad:
                        norm_string+=norm_val[i]+","
                    norm_string+=";VIOLATE:"
                    for i in vi:
                        norm_string+=norm_val[i]+","


                    example = {
                    "file_id": item,
                    "timestamp": start_time, 
                    "utterance": utt,
                    "norms": norm_string,
                    "label": cp,
                    "data_type":dt
                    }
                    
                    result.append(example)
    print(len(result))
    return result               
        
                



        

