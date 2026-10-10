"""Export a hardware-free viewer of original recordings and virtual commands."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil

import cv2


def optional_number(value):
    if value in ('', None):
        return None
    result = float(value)
    if not math.isfinite(result):
        return None
    return result


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_run(run, candidate=None):
    run = Path(run).resolve()
    csv_path = run/'onboard_log.csv'
    with csv_path.open() as stream:
        rows = list(csv.DictReader(stream))
    if not rows or [int(row['sample']) for row in rows] != list(range(len(rows))):
        raise ValueError('recorded CSV samples are not complete and consecutive')
    video = cv2.VideoCapture(str(run/'raw.mp4'))
    try:
        fps = video.get(cv2.CAP_PROP_FPS)
        count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        if not video.isOpened() or not math.isfinite(fps) or fps <= 0 or count != len(rows):
            raise ValueError('raw video and control CSV do not contain aligned frame counts')
    finally:
        video.release()
    derived = None
    if candidate is not None:
        candidate = Path(candidate).resolve()
        derived = json.loads((candidate/'commands.json').read_text())
        if len(derived) != len(rows) or [item['frame'] for item in derived] != list(range(len(rows))):
            raise ValueError('candidate replay commands do not align with the recorded run')
        summary = json.loads((candidate/'summary.json').read_text())
        if Path(summary['input_run']).resolve() != run:
            raise ValueError('candidate replay belongs to a different source run')
    frames = []
    for i, row in enumerate(rows):
        item = dict(sample=i, video_t=i/fps, capture_t=optional_number(row.get('timestamp_s')),
                    action=row.get('action'), reason=row.get('reason'),
                    pwm=[int(row.get(key) or 0) for key in ('front_left_pwm','rear_left_pwm',
                                                          'front_right_pwm','rear_right_pwm')],
                    confidence=optional_number(row.get('confidence')),
                    heading=optional_number(row.get('heading_error')),
                    semantic_age=optional_number(row.get('semantic_result_age_exact_s') or row.get('semantic_result_age_s')),
                    semantic_sequence=optional_number(row.get('semantic_sequence')),
                    yaw_rad=optional_number(row.get('imu_yaw_rad')) if row.get('imu_valid')=='True' else None,
                    imu_age=optional_number(row.get('imu_age_s')),
                    encoder=([optional_number(row.get('encoder_%d'%j)) for j in range(1,5)]
                             if row.get('encoder_valid')=='True' else None),
                    phase=row.get('stationary_phase') or None,
                    front_ratio=optional_number(row.get('semantic_front_boundary_ratio')),
                    right_exit=row.get('semantic_right_exit_observed')=='True', candidate=None)
        if derived is not None:
            changed=derived[i];state=changed.get('stationary_corner') or {}
            item['candidate']=dict(action=changed['action'],pwm=changed['pwm'],reason=changed['reason'],
                                   phase=state.get('stationary_phase'),
                                   front_ratio=state.get('semantic_front_boundary_ratio'),
                                   right_exit=state.get('semantic_right_exit_observed'),
                                   front_observed=state.get('semantic_front_observed'),
                                   right_exit_support=state.get('semantic_right_exit_support_ratio'))
        frames.append(item)
    result=dict(source_run=str(run), video_fps=fps, frames=frames,
                evidence_type='real-recording', limitations=[
                    '视频编码时间 video_t 与相机采集时间 capture_t 分开显示；按帧索引对应控制日志。',
                    '回放沿用历史画面，修改指令不会改变下一历史帧；它不能证明新的实车轨迹。',
                    '缺失或未同步的姿态、编码器保持未知；未从 PWM 虚构实际位置。'])
    return result


def export_bundle(output, runs, candidates=None, course_image=None, synthetic=None, viewer_path=None,
                  browser_media=None):
    output=Path(output).resolve()
    candidates=candidates or {}
    data=dict(schema_version=1,title='小车离线回放与转弯模拟',
              generated_at=datetime.now(timezone.utc).isoformat(),course_image=None,runs=[],synthetic=None)
    records={}
    for label,run in runs.items():
        if not label or not all(c.isalnum() or c in '-_' for c in label):
            raise ValueError('run ID must contain only letters, digits, dash or underscore')
        records[label]=read_run(run,candidates.get(label))
    display_media={}
    if browser_media:
        browser_media=Path(browser_media).resolve()
        for label,item in records.items():
            for name in ('raw.mp4','annotated.mp4'):
                original=Path(item['source_run'])/name
                if not original.is_file():
                    continue
                derived=browser_media/(label+'-'+name)
                video=cv2.VideoCapture(str(derived))
                try:
                    if (not video.isOpened()
                            or int(video.get(cv2.CAP_PROP_FRAME_COUNT)) != len(item['frames'])
                            or not math.isclose(video.get(cv2.CAP_PROP_FPS),item['video_fps'],abs_tol=1e-6)):
                        raise ValueError('browser media does not preserve recorded frame alignment: '+str(derived))
                finally:
                    video.release()
                display_media[(label,name)]=derived
    template=Path(viewer_path) if viewer_path else Path(__file__).with_name('viewer.html')
    if not template.is_file():
        raise ValueError('viewer template is missing')
    course_source=Path(course_image).resolve() if course_image else None
    if course_source is not None and not course_source.is_file():
        raise ValueError('course image is missing')
    synthetic_source=Path(synthetic).resolve() if synthetic else None
    synthetic_data=json.loads(synthetic_source.read_text()) if synthetic_source else None
    output.mkdir(parents=True,exist_ok=False);media=output/'media';media.mkdir()
    manifest=[]
    for label,item in records.items():
        run=Path(item['source_run'])
        item.update(id=label,label=label,raw_video='media/'+label+'-raw.mp4',annotated_video=None)
        for name in ('raw.mp4','annotated.mp4'):
            path=run/name
            if path.is_file():
                display_path=display_media.get((label,name),path)
                target=media/(label+'-'+name);target.symlink_to(display_path)
                if name=='annotated.mp4':item['annotated_video']='media/'+target.name
                record={'path':str(path),'sha256':sha256(path),'link':str(target.relative_to(output))}
                if display_path != path:
                    record.update(display_path=str(display_path),display_sha256=sha256(display_path),
                                  role='derived browser display copy; original remains replay input')
                manifest.append(record)
        if browser_media:
            item['limitations'].append('网页视频为保持帧数与帧率的派生显示副本；模型回放仍使用原始无损输入。')
        for name in ('onboard_log.csv','status.json','config.yaml','runtime_config.yaml',
                     'resolved_runtime_config.yaml','vehicle_profile.yaml','perspective_calibration.yaml'):
            path=run/name
            if path.is_file():manifest.append({'path':str(path),'sha256':sha256(path)})
        if label in candidates:
            for name in ('commands.json','summary.json','replay_config.yaml'):
                path=Path(candidates[label])/name
                if path.is_file():manifest.append({'path':str(path.resolve()),'sha256':sha256(path)})
        data['runs'].append(item)
    if course_source:
        source=course_source
        target=media/('course'+source.suffix.lower());target.symlink_to(source)
        data['course_image']='media/'+target.name
        manifest.append({'path':str(source),'sha256':sha256(source),'link':str(target.relative_to(output))})
    if synthetic_source:
        source=synthetic_source
        data['synthetic']=synthetic_data
        data['synthetic']['label']='SYNTHETIC 转弯流程验证，非实车整圈'
        manifest.append({'path':str(source),'sha256':sha256(source)})
    shutil.copyfile(template,output/'index.html')
    (output/'data.js').write_text('window.VEHICLE_SIM_DATA='+json.dumps(data,ensure_ascii=True)+';\n')
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    return data


def pairs(values):
    result={}
    for text in values:
        label,value=text.split('=',1)
        if label in result:raise ValueError('duplicate run ID: '+label)
        result[label]=value
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--run',action='append',default=[],required=True,help='ID=run directory')
    parser.add_argument('--candidate',action='append',default=[],help='ID=replay directory')
    parser.add_argument('--course-image')
    parser.add_argument('--synthetic',help='JSON containing labelled synthetic frames/summary')
    parser.add_argument('--browser-media',help='Directory of aligned ID-raw.mp4 / ID-annotated.mp4 display copies')
    args=parser.parse_args()
    data=export_bundle(args.output,pairs(args.run),pairs(args.candidate),args.course_image,args.synthetic,
                       browser_media=args.browser_media)
    print(json.dumps({'output':str(Path(args.output).resolve()),'runs':len(data['runs']),
                      'frames':sum(len(r['frames']) for r in data['runs'])}))


if __name__=='__main__':main()
