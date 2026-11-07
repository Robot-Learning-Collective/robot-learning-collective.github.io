'use strict';
// A source-backed example, not a claim to expose unrecorded model reasoning.
const loopExample=[
  {key:'orientation',label:'Report the problem',detail:'Returned model summary',text:'“I’ve flipped the object upside down!”',evidence:()=>sources.orientation.excerpt.replace(/^\*\*.*?\*\*\s*/,'').split('!')[0]+'!'},
  {key:'ik2',label:'Compute candidate poses',detail:'Python · local_ik.solve',text:'Two calls test three candidates each.',evidence:()=>sources.ik2.raw.arguments.command},
  {key:'correct',label:'Command the robot',detail:'Python · sim_client.act',text:'Send a computed left-arm joint target.',evidence:()=>sources.correct.raw.arguments.command},
  {key:'correct',label:'Receive feedback',detail:'Tool result · image + telemetry',text:'A new camera image arrives; telemetry reports stalled motion.',evidence:()=>sources.correct.raw.result.content.filter(c=>c.type==='text').map(c=>c.text).join('\n').split('\n').find(line=>line.includes('status=stalled'))}
];
let loopSelected=0;
function showLoopStep(index,{select=false}={}){
  loopSelected=index;const item=loopExample[index],r=sources[item.key];
  document.querySelectorAll('.loopStep').forEach((b,i)=>b.setAttribute('aria-pressed',String(i===index)));
  el('loopEvidenceLabel').textContent=`${item.detail} · ${time(index===3?r.end:r.t)} wall · ${index===0?'verbatim excerpt':index===3?'verbatim result line':'recorded command'}`;
  el('loopEvidenceText').textContent=item.evidence();
  if(select)chooseMoment(0,{source:r});
}
loopExample.forEach((item,index)=>{
  const button=make('button',undefined,'loopStep');
  button.append(make('small',`0${index+1} / ${item.detail}`),make('strong',item.label),make('span',item.text));
  button.addEventListener('click',()=>showLoopStep(index,{select:true}));el('loopSteps').append(button);
});
function revealTarget(target){
  for(let node=target;node;node=node.parentElement)if(node.tagName==='DETAILS')node.open=true;
}
el('loopSource').addEventListener('click',()=>{
  chooseMoment(0,{source:sources[loopExample[loopSelected].key]});
  revealTarget(el('sourceDetail'));el('selectedSource').scrollIntoView({behavior:'smooth',block:'start'});
});
// Reveal disclosure ancestors before existing source links scroll into view.
document.addEventListener('click',event=>{
  const link=event.target.closest('a[href^="#"]');
  if(link){const target=document.getElementById(link.getAttribute('href').slice(1));if(target)revealTarget(target);}
  if(event.target.closest('#segmentQuote'))revealTarget(el('selectedSource'));
},true);
function revealHashTarget(){const target=document.getElementById(location.hash.slice(1));if(target)revealTarget(target);const params=new URLSearchParams(location.hash.slice(1));if(params.has('evidence')||params.has('event'))revealTarget(el('selectedSource'));}
window.addEventListener('hashchange',revealHashTarget);
showLoopStep(0);revealHashTarget();
