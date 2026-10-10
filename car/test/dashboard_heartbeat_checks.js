// Pure JS fault injection; no browser/network/hardware. Caller supplies asset.
async function verifyDashboardHeartbeat(source) {
 const checks=[];
 function check(ok,name){if(!ok)throw new Error(name);checks.push(name);}
 function fixture(){
  const nodes={},intervals=[],timers=[],calls=[];let responder=()=>new Promise(()=>{});
  const node=id=>nodes[id]||(nodes[id]={dataset:{},textContent:'',disabled:false,hidden:false,
   classList:{add(){},remove(){},toggle(){}},parentElement:{classList:{add(){},toggle(){}}},
   replaceChildren(){},add(){},addEventListener(event,fn){this[event]=fn;}});
  const fetcher=(url,options)=>{calls.push({url,options});return responder(url,options);};
  const script=new Function('document','fetch','setInterval','setTimeout','clearTimeout',
   'AbortController','performance','Option',source+`
   return {own:id=>ownedSession=id,owner:()=>ownedSession,disconnect:disconnected,
    render:renderDriveControl};`);
  const app=script({getElementById:node},fetcher,(fn,ms)=>intervals.push({fn,ms}),
   (fn,ms)=>{timers.push({fn,ms});return timers.length;},()=>{},
   class{constructor(){this.signal={};}abort(){}},{now:()=>0},class{});
  return {app,node,calls,timers,respond:fn=>responder=fn,
   heartbeat:intervals.find(x=>x.ms===700).fn};
 }
 function reply(status,value){return {ok:status===200,status,json:async()=>value};}
 {
  const f=fixture();f.app.own('a');f.app.disconnect();
  check(f.app.owner()==='a','monitor GET failure preserves control heartbeat ownership');
  f.respond(async()=>{throw new Error('temporary network failure');});await f.heartbeat();
  check(f.app.owner()==='a','single heartbeat network failure retains owner for retry');
  f.respond(async()=>reply(503,{error:'temporarily unavailable'}));await f.heartbeat();
  check(f.app.owner()==='a','HTTP 503 retains owner for retry');
  f.respond(async()=>reply(200,{session_id:'a',running:true}));await f.heartbeat();
  check(f.calls.filter(x=>x.url.endsWith('/heartbeat')).length===3,'heartbeats continue after transient failures');
  check(f.timers.slice(1).every(x=>x.ms===1000),'heartbeat timeout permits retry inside original three-second lease');
  f.respond(async()=>reply(409,{error:'session is stopping'}));await f.heartbeat();
  check(f.app.owner()===null,'explicit session rejection ends heartbeat');
 }
 {
  const f=fixture();f.app.own('a');let complete;
  f.respond(()=>new Promise(resolve=>complete=resolve));
  const pending=f.heartbeat();await f.heartbeat();
  check(f.calls.filter(x=>x.url.endsWith('/heartbeat')).length===1,'no overlapping heartbeat requests');
  f.app.own('b');complete(reply(409,{error:'old session stopping'}));await pending;
  check(f.app.owner()==='b','late old rejection cannot clear new session ownership');
  let fail;f.respond(()=>new Promise((_,reject)=>fail=reject));const old=f.heartbeat();
  f.app.own(null);fail(new Error('network failure after end click'));await old;
  check(f.app.owner()===null,'late network failure cannot resurrect ended ownership');
 }
 {
  const f=fixture();f.app.own('a');f.app.render({drive_control:{available:true,session_id:'a',running:false,
   state:'finished',reason:'panel control lease expired'},drive_runs:[]});
  check(f.app.owner()===null,'confirmed finished session ends heartbeat');
  check(f.node('driveControlHint').textContent.includes('控制心跳超时'),'panel shows actual lease exit reason');
 }
 return {passed:checks.length,checks,scope:'DOM/fetch/timers simulated; no network/hardware, real asset evaluated'};
}
