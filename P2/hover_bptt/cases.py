import copy,json
from pathlib import Path

def evaluation_cases():return json.loads((Path(__file__).parent/'configs/cases.json').read_text(encoding='utf-8'))
def training_cases():
    return[dict(copy.deepcopy(c),trial=r,actual_body_drag_scale=scale,training_condition_index=i)
        for r,scale in enumerate([1.,1.0125,1.025])for i,c in enumerate(evaluation_cases())]
