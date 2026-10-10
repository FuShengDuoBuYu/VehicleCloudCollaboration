'use strict';
const el = id => document.getElementById(id);
const labels={hold:'保持停车',online:'在线',live:'实时',stale:'数据过期',not_integrated:'未接入',not_started:'未启动',stopped:'已结束',disabled:'未启用',error:'异常',busy:'设备占用',waiting:'等待数据',completed:'已返回',pending:'请求中',requesting:'请求中',timeout:'超时',failed:'失败',unknown:'未知',idle:'等待触发',disconnected:'连接中断'};
const names={battery:'电池',camera:'RGB 相机',lidar:'激光雷达',depth:'深度相机',imu:'IMU',attitude:'姿态',encoder:'编码器'};
const text=v=>v===null||v===undefined||v===''?'—':String(v);
const number=(v,n=2)=>typeof v==='number'&&Number.isFinite(v)?v.toFixed(n):'—';
const age=v=>typeof v==='number'?`${number(v,1)} s 前`:'无采样';
function badge(node,status){node.className='badge '+(Object.hasOwnProperty.call(labels,status)?status:'unknown');node.textContent=labels[status]||text(status);}
function pairs(id,rows){const node=el(id);node.replaceChildren();rows.forEach(([key,value])=>{const dt=document.createElement('dt'),dd=document.createElement('dd');dt.textContent=key;dd.textContent=text(value);node.append(dt,dd);});}
function vector(v){return Array.isArray(v)?v.map(x=>number(x,2)).join(' / '):'—';}
function sensorValue(name,item){const v=item.value||{};if(!item.value)return item.reason||'尚无数据';switch(name){case'battery':return `${number(v.voltage_v,1)} V · ${item.status==='online'&&Number.isFinite(v.percentage)?'估算电量 '+number(v.percentage,0)+'%':'电量未知'}`;case'camera':return `${text(v.source)}${v.width?' · '+v.width+' × '+v.height:''}${v.fps?' · '+number(v.fps,1)+' Hz':''}`;case'lidar':return `${number(v.fps,1)} Hz · ${text(v.valid_points)} 有效点 · 最近 ${number(v.nearest_m)} m`;case'depth':return `${number(v.fps,1)} Hz · 有效率 ${number(v.valid_fraction*100,1)}% · 中位 ${number(v.median_m)} m`;case'imu':return `角速度 ${vector(v.gyro_rad_s)} rad/s`;case'attitude':return `${vector(v.radians)} rad`;case'encoder':return Array.isArray(v.native_ticks)?v.native_ticks.join(' / '):'—';}}
let lastState=null,lastSuccess=0,fetching=false,frameAt=0;
let driveBusy=false,ownedSession=null,heartbeatBusy=false;
function renderDriveControl(state){
 const c=state.drive_control||{},active=!!c.running;
 el('startDrive').disabled=driveBusy||!c.available||active;
 el('stopDrive').disabled=!c.available||!active;
 el('startDrive').textContent=c.mode==='dry-run'?'开始预演（车轮不动）':'开始自动驾驶';
 const names={idle:'待机',starting:'准备感知与记录',running:c.mode==='dry-run'?'正在预演（车轮不动）':'自动驾驶中',stopping:'正在停车并保存记录',finished:'已结束',error:'运行异常',unavailable:'车端启停程序尚未就绪'};
 el('driveControlHint').textContent=(names[c.state]||'启停程序尚未就绪')+(c.runtime_limit_seconds===0?' · 无总时限，手动结束':'')+(c.session_id?' · '+c.session_id.slice(0,8):'');
 if(['stopping','finished','error'].includes(c.state)&&c.reason){
  const reasons={'panel control lease expired':'控制心跳超时，已请求停车','operator clicked end':'手动结束','panel supervisor closing':'启停程序关闭','supervisor interrupted':'启停程序中断'};
  el('driveControlHint').textContent+=' · '+(reasons[c.reason]||c.reason);
 }
 el('replayDrive').hidden=active||!c.run_archive||!c.session_id;
 if(c.session_id)el('replayDrive').href='/replay?session_id='+encodeURIComponent(c.session_id);
 const history=el('driveHistory'),runs=state.drive_runs||[];
 if(history.dataset.runs!==JSON.stringify(runs)){
  history.dataset.runs=JSON.stringify(runs);history.replaceChildren(new Option('选择一次运行回放',''));
  runs.forEach(run=>history.add(new Option(run.run_id,run.session_id)));
 }
 history.disabled=active;
 // A poll begun before the Start response can still carry the old idle state.
 // Read-only monitor failures do not cancel the independent control heartbeat.
 // Keep ownership until this session finishes or control explicitly rejects it.
 if(ownedSession&&!active&&c.session_id===ownedSession)ownedSession=null;
 if(c.available){el('notice').textContent=active?(c.mode==='dry-run'?'正在预演控制与记录（车轮不动）':'由最新 YOLOPv2 道路结果决定下一步动作 · 正在按次记录'):'点击开始运行；点击结束停车并保存记录。';}
}
async function driveIntent(action,body){
 const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),action==='heartbeat'?1000:3000);
 try{const response=await fetch('/api/drive/'+action,{method:'POST',headers:{'Content-Type':'application/json','X-Vehicle-Control':'panel'},body:JSON.stringify(body),signal:controller.signal});const value=await response.json();if(!response.ok)throw Object.assign(new Error(value.error||'启停请求失败'),{status:response.status});return value;}finally{clearTimeout(timer);}
}
el('startDrive').addEventListener('click',async()=>{
 if(driveBusy)return;driveBusy=true;el('startDrive').disabled=true;
 try{const c=await driveIntent('start',{request_id:Date.now().toString(16)+Math.random().toString(16).slice(2)});ownedSession=c.session_id;lastState={...(lastState||{}),drive_control:c};renderDriveControl(lastState);}
 catch(error){el('driveControlHint').textContent=error.message;}finally{driveBusy=false;}
});
el('stopDrive').addEventListener('click',async()=>{
 const c=(lastState||{}).drive_control||{};ownedSession=null;
 try{const result=await driveIntent('stop',{session_id:c.session_id});lastState={...(lastState||{}),drive_control:result};renderDriveControl(lastState);}catch(error){el('driveControlHint').textContent='结束请求未确认：'+error.message;}
});
el('driveHistory').addEventListener('change',()=>{const id=el('driveHistory').value;if(id)window.open('/replay?session_id='+encodeURIComponent(id),'_blank','noopener');});
setInterval(async()=>{
 if(!ownedSession||heartbeatBusy)return;
 const session=ownedSession;heartbeatBusy=true;
 try{
  const c=await driveIntent('heartbeat',{session_id:session});
  if(ownedSession===session&&(c.session_id!==session||!c.running||c.state==='stopping'))ownedSession=null;
 }catch(error){
  if(ownedSession!==session)return;
  if(error.status===403||error.status===409){
   ownedSession=null;el('driveControlHint').textContent='控制会话已结束或被拒绝，等待停车确认';
  }else{
   // Retry transport/temporary server failures. The original three-second
   // vehicle lease still expires independently; no session is auto-started.
   el('driveControlHint').textContent='控制心跳暂未确认，正在重试；持续失联时车端停车';
  }
 }finally{heartbeatBusy=false;}
},700);
const imagePending={},imageErrors={};
function updateImage(id,url,live){
 const node=el(id);node.parentElement.classList.toggle('stale',!live);if(!url||imagePending[id])return;
 const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),1500);imagePending[id]=true;
 // Decode offscreen, then swap a complete image. Failed requests preserve
 // the last good frame rather than turning the displayed image black.
 (async()=>{let next=null;try{
  const response=await fetch(url+'?t='+Date.now(),{cache:'no-store',signal:controller.signal});
  if(!response.ok)throw new Error('image HTTP '+response.status);
  next=URL.createObjectURL(await response.blob());const candidate=new Image();candidate.src=next;await candidate.decode();
  delete imageErrors[id];const previous=node.dataset.objectUrl;node.src=next;node.dataset.objectUrl=next;next=null;
  if(previous)URL.revokeObjectURL(previous);
 }catch(error){imageErrors[id]=true;node.parentElement.classList.add('stale');if(id==='camera'){el('frameCover').hidden=false;el('frameCover').textContent='等待完整新帧 · 保留上一次画面';}}
 finally{if(next)URL.revokeObjectURL(next);clearTimeout(timer);imagePending[id]=false;}})();
}
function radar(sensor){const canvas=el('radar'),ctx=canvas.getContext('2d'),w=canvas.width,h=canvas.height;ctx.clearRect(0,0,w,h);ctx.strokeStyle='#d0dfe1';[35,70,100].forEach(r=>{ctx.beginPath();ctx.arc(w/2,h/2,r,0,Math.PI*2);ctx.stroke();});ctx.beginPath();ctx.moveTo(w/2,0);ctx.lineTo(w/2,h);ctx.moveTo(0,h/2);ctx.lineTo(w,h/2);ctx.stroke();ctx.fillStyle='#176d63';ctx.beginPath();ctx.moveTo(w/2,h/2-6);ctx.lineTo(w/2-4,h/2+4);ctx.lineTo(w/2+4,h/2+4);ctx.fill();const v=sensor.value||{};if(sensor.status==='online'&&Array.isArray(v.xy_m)){ctx.fillStyle='#078a70';v.xy_m.slice(0,360).forEach(p=>{if(Array.isArray(p)&&p.length===2&&p.every(Number.isFinite)){const x=w/2-p[1]*30,y=h/2-p[0]*30;if(x>=0&&x<=w&&y>=0&&y<=h)ctx.fillRect(x,y,2,2);}});}el('radarHint').textContent=`${labels[sensor.status]||'未知'} · 量程视图约 ±3 m · 传感器坐标，未经车身外参标定`;}
function modelStatus(state){const runtime=state.runtime||{},yolo=(runtime.diagnostics||{}).yolopv2_fusion||{},row=runtime.last_result||{},auto=state.autonomy||{};if((state.drive_control||{}).state==='starting'||auto.status!=='live'||!yolo.enabled)return 'waiting';if(yolo.error)return 'error';const value=row.semantic_result_age_s;const maximum=typeof yolo.max_result_age_seconds==='number'?yolo.max_result_age_seconds:.3;return typeof value==='number'&&Number.isFinite(value)&&value>=0&&value+Math.max(0,auto.age_s||0)<=maximum?'online':'stale';}
function render(state){lastState=state;const sensors=state.sensors||{},runtime=state.runtime||{},row=runtime.last_result||{},auto=state.autonomy||{},cloud=state.cloud||{},nav=state.navigation||{},sys=state.system||{},yolo=(runtime.diagnostics||{}).yolopv2_fusion||{};const live=auto.status==='live';badge(el('connection'),'online');el('clock').textContent=new Date(state.server_time*1000).toLocaleTimeString('zh-CN',{hour12:false});el('host').textContent=`${text(sys.host)} · ${location.host} · 服务运行 ${number(sys.uptime_s/3600,1)} h（主机启动时间）`;el('notice').classList.remove('alert');el('notice').textContent=live?'只读监控 · 正在接收车端运行数据 · 提案、感知置信度不等于实际执行或安全保证':'只读监控 · '+(labels[auto.status]||'等待运行')+' · 未运行时显示已接入传感器状态';
const battery=sensors.battery||{},batteryValue=battery.value||{};el('batteryPercentage').textContent=(battery.status==='online'?number(batteryValue.percentage,0):'—')+' %';el('voltage').textContent=(battery.status==='online'?number(batteryValue.voltage_v,1):'—')+' V';el('batteryHint').textContent=(labels[battery.status]||'未知')+' · '+age(battery.age_s)+' · 10–12 V 线性估算';el('drive').textContent=live&&runtime.mode==='dry-run'?'感知监控':(labels[auto.status]||text(auto.status));el('driveHint').textContent=`${text(runtime.mode)} · ${age(auto.age_s)}`;el('yolo').textContent=modelStatus(state)==='online'?text(yolo.latest_precision).toUpperCase():labels[modelStatus(state)];el('yoloHint').textContent=`推理 ${number(row.semantic_inference_ms,0)} ms · 已完成 ${text(yolo.completed_frames)} 帧 · 画面目标 100 ms`;const count=Object.values(sensors).filter(s=>s.status==='online').length;el('sensorCount').textContent=count+' / '+Object.keys(names).length;el('sensorHint').textContent='在线采样数 · 设备枚举不代表正常采集';el('cloudState').textContent=labels[cloud.status]||text(cloud.status);el('cloudHint').textContent=cloud.source_status==='live'?`事件 ${text(cloud.event_id)}`:'当前无实时云端事件';
const rows=el('sensorRows');rows.replaceChildren();Object.entries(names).forEach(([key,name])=>{const item=sensors[key]||{status:'not_integrated'},tr=document.createElement('tr'),a=document.createElement('td'),b=document.createElement('td'),c=document.createElement('td'),pill=document.createElement('span'),stamp=document.createElement('div');a.textContent=name;badge(pill,item.status);stamp.className='age';stamp.textContent=age(item.age_s);b.append(pill,stamp);c.textContent=sensorValue(key,item);tr.append(a,b,c);rows.append(tr);});radar(sensors.lidar||{status:'not_integrated'});
el('confidence').textContent=live?number(row.confidence,3):'—';el('modelAge').textContent=live?number(row.semantic_result_age_s??row.semantic_age_s,3)+' s':'—';el('steering').textContent=live?number(row.steering,3):'—';el('action').textContent=live?text(row.action):'—';el('reason').textContent=live?text(row.reason):'暂无实时驾驶判断';
const frame=(state.frames||{}).annotated||{},bird=(state.frames||{}).birdeye||{},preparing=(state.drive_control||{}).state==='starting';badge(el('frameStatus'),preparing?'waiting':frame.live?'online':frame.available?'stale':'waiting');el('frameCover').hidden=!!frame.live&&!preparing&&!imageErrors.camera;el('frameCover').textContent=preparing?'正在切换相机、预热模型并准备记录 · 此时保持停车':imageErrors.camera?'等待完整新帧 · 保留上一次画面':frame.available?'历史画面 · '+age(frame.age_seconds):'等待运行时发布相机与模型画面';badge(el('cloudBadge'),cloud.status);const result=cloud.result||{};pairs('cloudDetails',[['状态时效',cloud.source_status==='live'?'实时':'未接入或历史状态'],['触发原因',cloud.trigger_reason],['事件 / 请求',text(cloud.event_id)+' / '+text(cloud.request_id)],['模型',cloud.model],['请求耗时',cloud.latency_ms===undefined?'—':number(cloud.latency_ms,0)+' ms'],['场景',result.scene_summary],['风险 / 建议',text(result.risk_level)+' / '+text(result.recommendation)],['建议依据',result.reason],['不确定性',Array.isArray(result.uncertainties)?result.uncertainties.join('；'):'—'],['云端置信度','接口未提供数值置信度'],['车端仲裁',typeof cloud.arbitration==='object'?JSON.stringify(cloud.arbitration):cloud.arbitration]]);
badge(el('goalBadge'),nav.status);pairs('goalDetails',[['任务',nav.task],['最终目的地',typeof nav.destination==='object'&&nav.destination?JSON.stringify(nav.destination):nav.destination],['路线约束',nav.route_constraint],['局部前视目标',nav.local_target?JSON.stringify(nav.local_target):'未发布（与最终目的地区分）'],['说明',nav.reason]]);const gate=runtime.motion_gate||{},watch=runtime.watchdog||{};pairs('controlDetails',[['模式',runtime.mode],['运动门禁',live?(gate.ready?'就绪':'未放行'):'无实时状态'],['停止原因',(runtime.termination||{}).reason||row.reason],['四轮 PWM 提案',live?[row.front_left_pwm,row.rear_left_pwm,row.front_right_pwm,row.rear_right_pwm].map(text).join(' / '):'—'],['看门狗',live?JSON.stringify(watch):'无实时状态'],['运行 ID',auto.run_id]]);
const objects=el('objects');objects.replaceChildren();const detections=Array.isArray(yolo.detections)?yolo.detections:[];badge(el('objectsBadge'),modelStatus(state));if(modelStatus(state)!=='online')objects.textContent='暂无有效实时检测结果 · '+labels[modelStatus(state)];else if(!detections.length)objects.textContent='本帧未报告目标（不代表道路无障碍）';else detections.slice(0,100).forEach(d=>{const div=document.createElement('div');div.className='object';const label=document.createElement('span'),score=document.createElement('span');label.textContent=text(d.label??d.class_name??d.class_id);score.textContent='检测置信度 '+number(d.confidence??d.score,3);div.append(label,score);objects.append(div);});
pairs('systemDetails',[['内存可用 / 总计',number(sys.memory_available_bytes/2**30,2)+' / '+number(sys.memory_total_bytes/2**30,2)+' GiB'],['磁盘可用',number(sys.disk_free_bytes/2**30,1)+' GiB'],['系统负载',vector(sys.load_average)],['温度',Object.entries(sys.temperatures_c||{}).map(([k,v])=>`${k} ${number(v,1)} °C`).join(' · ')]]);el('source').textContent='数据源 '+text(auto.source_directory);el('raw').textContent=JSON.stringify(state,null,2);renderDriveControl(state);
}
setInterval(()=>{if(!lastState||Date.now()-lastSuccess>1500)return;const frame=(lastState.frames||{}).annotated||{},bird=(lastState.frames||{}).birdeye||{},depth=(lastState.sensors||{}).depth||{};
 updateImage('camera',frame.available?'/api/latest-frame.jpg':null,frame.live);updateImage('birdeye',bird.available?'/api/latest-birdeye.jpg':null,bird.live);
 // Depth is a separate sensor; do not request the same slower image at 10 Hz.
 if(Date.now()-frameAt>=300){frameAt=Date.now();updateImage('depth',depth.status==='online'?'/api/depth.jpg':null,depth.status==='online');el('depthHint').textContent=labels[depth.status]||'等待深度图';}
 el('birdHint').textContent=bird.live?'实时鸟瞰':bird.available?'历史鸟瞰 · '+age(bird.age_seconds):'等待鸟瞰图';},100);
function disconnected(){badge(el('connection'),'disconnected');el('startDrive').disabled=true;el('batteryPercentage').textContent='— %';el('voltage').textContent='— V';el('batteryHint').textContent='连接中断 · 电量未知';el('notice').classList.add('alert');el('notice').textContent='监控数据连接中断 · 下方为最后收到的数据；驾驶控制心跳独立检测，不能据此判断小车当前状态';['camera','birdeye','depth'].forEach(id=>el(id).parentElement.classList.add('stale'));el('frameCover').hidden=false;el('frameCover').textContent='连接中断 · 非实时画面';}
async function refresh(){if(fetching)return;fetching=true;const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),3000),began=performance.now();try{const response=await fetch('/api/state',{cache:'no-store',signal:controller.signal});if(!response.ok)throw new Error('HTTP '+response.status);render(await response.json());lastSuccess=Date.now();}catch(error){disconnected();}finally{clearTimeout(timer);fetching=false;setTimeout(refresh,Math.max(0,100-(performance.now()-began)));}}
setInterval(()=>{if(lastSuccess&&Date.now()-lastSuccess>3500)disconnected();},500);
refresh();
