"""Contract selection is independent of transport; outputs never authorize motion."""
from .schema import SCENE_SCHEMA, SYSTEM_PROMPT, validate_scene

FEATURES = ['right_arrow','left_arrow','straight_arrow','crosswalk','parking_sign',
            'boundary_line','obstacle','occlusion','blur','horn_sign']
OBSERVATION_SCHEMA = {'type':'object','additionalProperties':False,
    'required':['features','uncertain','advice','reason'], 'properties':{
        'features':{'type':'array','uniqueItems':True,'items':{'type':'string','enum':FEATURES}},
        'uncertain':{'type':'boolean'},'advice':{'type':'string','enum':['stop','wait','observe']},
        'reason':{'type':'string','minLength':1,'maxLength':10}}}
OBSERVATION_PROMPT = (
    '你是低速小车视觉感知助手，只分析当前图片，仅输出紧凑JSON实例，不输出Schema或Markdown。'
    '格式{"features":[],"uncertain":true,"advice":"observe","reason":"简短依据"}。'
    'features列全实际可见特征，仅用right_arrow/left_arrow/straight_arrow/crosswalk/parking_sign/'
    'boundary_line/obstacle/occlusion/blur/horn_sign。复合箭头分别列所有方向，弯曲箭头沿主干和尖端判断方向。'
    '模糊、遮挡或通行无法确定时uncertain=true，不猜测距离、不把远处墙面直接当作堵路。'
    'advice只可stop/wait/observe，此接口只提供感知，不授权运动。reason不超过10个汉字。')


def validate_observation(value):
    if not isinstance(value,dict) or set(value)!=set(OBSERVATION_SCHEMA['required']):
        raise ValueError('observation fields mismatch')
    features=value['features']
    if not isinstance(features,list) or any(not isinstance(x,str) or x not in FEATURES for x in features):
        raise ValueError('invalid observation features')
    if len(set(features))!=len(features): raise ValueError('duplicate observation features')
    if type(value['uncertain']) is not bool: raise ValueError('invalid uncertainty')
    if not isinstance(value['advice'],str) or value['advice'] not in {'stop','wait','observe'}:
        raise ValueError('invalid observation advice')
    reason=value['reason']
    if not isinstance(reason,str) or not reason.strip() or len(reason)>10: raise ValueError('invalid observation reason')
    try: reason.encode('utf-8')
    except UnicodeError as error: raise ValueError('invalid observation Unicode') from error
    return value


CONTRACTS = {
    'road-observation-fast-v1': (OBSERVATION_SCHEMA,OBSERVATION_PROMPT,validate_observation,'road-observation-prompt-v1'),
    'road-scene-v1': (SCENE_SCHEMA,SYSTEM_PROMPT,validate_scene,'road-scene-prompt-v1')}


def get_contract(name):
    if name not in CONTRACTS: raise ValueError('unsupported cloud contract')
    return CONTRACTS[name]
