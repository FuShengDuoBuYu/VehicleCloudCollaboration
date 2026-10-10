'use strict';
const el=id=>document.getElementById(id);
const session=new URLSearchParams(location.search).get('session_id');
let frames=[],index=0,playing=false,timer=null,generation=0;
const num=(value,n=3)=>typeof value==='number'&&Number.isFinite(value)?value.toFixed(n):'未知';
function pairs(rows){const node=el('readings');node.replaceChildren();rows.forEach(([key,value])=>{const a=document.createElement('dt'),b=document.createElement('dd');a.textContent=key;b.textContent=value;node.append(a,b);});}
function endpoint(kind,view){return '/api/drive/'+kind+'?session_id='+encodeURIComponent(session)+'&sample='+index+(view?'&view='+view:'');}
async function render(){
 if(!frames.length)return;const current=++generation,row=frames[index];
 el('scrub').value=index;el('sample').textContent=(index+1)+' / '+frames.length;el('time').textContent=num(row.timestamp_s,2)+' s';
 ['raw','annotated','depth'].forEach(view=>{el(view).onerror=()=>{el(view).parentElement.classList.add('stale');if(view==='depth')el('depthHint').textContent='本帧没有同步深度记录';};el(view).onload=()=>{el(view).parentElement.classList.remove('stale');if(view==='depth')el('depthHint').textContent='已保存的原始深度生成色图；零值保持缺失。';};el(view).src=endpoint('frame.jpg',view);});
 pairs([['当前动作',row.action||'未知'],['原因',row.reason||'未知'],['四轮 PWM 日志',[row.front_left_pwm,row.rear_left_pwm,row.front_right_pwm,row.rear_right_pwm].map(v=>num(v,0)).join(' / ')],['道路方向误差',num(row.heading_error)],['YOLO 结果年龄',num(row.semantic_result_age_exact_s)+' s'],['IMU 航向',row.imu_valid?num(row.imu_yaw_rad)+' rad':'缺失或过期'],['编码器原生 ticks',row.encoder_valid?[1,2,3,4].map(i=>num(row['encoder_'+i],0)).join(' / '):'缺失或过期'],['电压',row.telemetry?.motion?.valid?num(row.telemetry.motion.value?.battery_v,1)+' V':'缺失或过期'],['角速度',row.telemetry?.imu?.valid?(row.telemetry.imu.value?.gyro_rad_s||[]).map(v=>num(v)).join(' / ')+' rad/s':'缺失或过期'],['加速度',row.telemetry?.imu?.valid?(row.telemetry.imu.value?.accel_m_s2||[]).map(v=>num(v)).join(' / ')+' m/s²':'缺失或过期'],['控制阶段',row.stationary_phase||'未知']]);
 el('row').textContent=JSON.stringify(row,null,2);
 const canvas=el('radar'),ctx=canvas.getContext('2d');ctx.clearRect(0,0,360,250);
 try{const response=await fetch(endpoint('scan'),{cache:'no-store'});if(!response.ok)throw new Error();const scan=await response.json();if(current!==generation)return;ctx.fillStyle='#078a70';scan.xy_m.forEach(([x,y])=>{const px=180-y*35,py=125-x*35;if(px>=0&&px<360&&py>=0&&py<250)ctx.fillRect(px,py,2,2);});ctx.fillStyle='#172f37';ctx.fillRect(177,122,6,6);el('radarHint').textContent='完整扫描 '+scan.points+' 点 · 采样早于控制 '+num(scan.age_s,3)+' s · 传感器坐标';}
 catch(error){if(current===generation)el('radarHint').textContent='本帧没有同步雷达记录';}
}
function pause(){playing=false;clearTimeout(timer);el('play').textContent='播放';}
function advance(){if(!playing)return;if(index>=frames.length-1){pause();return;}const gap=(frames[index+1].timestamp_s-frames[index].timestamp_s)*1000;timer=setTimeout(()=>{index++;render();advance();},Math.max(10,gap));}
el('play').addEventListener('click',()=>{if(playing){pause();return;}if(index===frames.length-1)index=0;playing=true;el('play').textContent='暂停';render();advance();});
el('scrub').addEventListener('input',()=>{pause();index=Number(el('scrub').value);render();});
el('previous').addEventListener('click',()=>{pause();index=Math.max(0,index-1);render();});
el('next').addEventListener('click',()=>{pause();index=Math.min(frames.length-1,index+1);render();});
(async()=>{try{const response=await fetch('/api/drive/replay-data?session_id='+encodeURIComponent(session));if(!response.ok)throw new Error('本次记录尚未完整保存或当前仍在驾驶');const data=await response.json();frames=data.frames;el('run').textContent='运行 '+data.run_id+' · '+frames.length+' 帧';el('scrub').max=frames.length-1;['play','previous','next'].forEach(id=>el(id).disabled=false);render();}catch(error){el('notice').textContent=error.message;el('notice').classList.add('alert');}})();
