'use strict';
const traceKinds=new Set(['model','tool','robot','context']);
const kindGroup=r=>r.kind==='summary'?'model':r.kind;
function inspectRecord(r,{push=true}={}) {
  const index=moments.findIndex(m=>r.t>=m.start-.01&&r.t<=m.end);
  if(index>=0)chooseMoment(index,{push,source:r});else selectSource(r,{push});
}
function syncStorySelection(r) {
  setClockInspection(r.t,r.frame);
  document.querySelectorAll('.bandEvent').forEach(n=>{n.classList.toggle('active',n.dataset.moment===moments[active].id);n.setAttribute('aria-current',String(n.dataset.moment===moments[active].id));});
  const row=[...el('recordList').children].find(n=>n.dataset.source===r.id);
  if(row)el('recordList').scrollTop=row.offsetTop-el('recordList').offsetTop-80;
}
function renderRecordList() {
  const query=el('traceSearch').value.trim().toLowerCase();
  const records=TRACE.records.filter(r=>traceKinds.has(kindGroup(r))&&(!query||`${r.id} ${r.title} ${r.excerpt||''} ${JSON.stringify(r.raw.arguments||{})}`.toLowerCase().includes(query)));
  el('recordCount').textContent=`${records.length} / ${TRACE.records.length} activity records`;
  el('recordList').replaceChildren();
  for(const r of records){const b=make('button',undefined,`recordRow ${kindGroup(r)}`);b.dataset.source=r.id;b.setAttribute('aria-pressed',String(r.id===selected.id));b.classList.toggle('selected',r.id===selected.id);
    const title=recordPreview(r);
    b.append(make('small',`${time(r.t)} · ${r.kind}`),make('span',title));b.addEventListener('click',()=>inspectRecord(r));el('recordList').append(b);
  }
  if(!records.length)el('recordList').append(make('p','No records match these filters.','figcaption'));
  document.querySelectorAll('#traceLanes .lane').forEach(n=>n.hidden=!traceKinds.has(n.dataset.kind));
}
function buildStateLanes() {
  for(const [key,label] of [['left','Left grasp'],['right','Right grasp'],['goal','Attached']]){
    const row=make('div',undefined,'lane stateLane'),track=make('div',undefined,'stateTrack');row.append(make('span',label),track);
    let first=TRACE.states[0];
    const add=end=>{const r=stateRecord(first.step);byId.set(r.id,r);const value=first[key],b=make('button',undefined,`stateSpan ${value===null?'unmeasured':value?'stateTrue':'stateFalse'}`);b.dataset.source=r.id;b.style.left=`${first.t/TRACE.duration*100}%`;b.style.width=`${Math.max(.12,(end-first.t)/TRACE.duration*100)}%`;b.title=`${label}: ${value===null?'unmeasured':value?'true':'false'} · ${time(first.t)}–${time(end)} wall`;b.setAttribute('aria-label',b.title);b.addEventListener('click',()=>inspectRecord(r));track.append(b);};
    for(const s of TRACE.states.slice(1))if(s[key]!==first[key]){add(s.t);first=s;}add(TRACE.duration);el('stateLanes').append(row);
  }
}
function renderAlignmentLadder() {
  el('diagramTitle').textContent='Alignment attempts';
  const definitions=[
    [1300.41004,'Approach','Turn base toward the wall'],[1317.70213,'Approach','Move to a stand-off from the nail'],[1421.55138,'Approach','Raise poster with a feasible joint target'],[1440.29039,'Approach','Move toward the nail; reserve clearance'],
    [1459.87793,'Seating probes','40 mm normal approach'],[1476.25931,'Seating probes','25 mm normal approach'],[1491.89893,'Seating probes','Another 25 mm approach'],[1512.31812,'Seating probes','20 mm seating probe'],[1525.18191,'Seating probes','Lower 10 mm while advancing'],[1538.72266,'Seating probes','15 mm seating probe'],[1551.86561,'Seating probes','Another 15 mm probe'],
    [1581.54137,'Reassessment','Back off 15 mm and hold joint targets'],[1594.42607,'Reassessment','Lower 20 mm with a short advance'],[1620.89799,'Reassessment','Probe at the revised height'],[1655.12795,'Reassessment','Retreat 40 mm; reset wrist pitch'],
    [1763.94673,'Fresh geometry','Lower poster 180 mm to expose nail'],[1808.41781,'Fresh geometry','Apply the refined alignment target']
  ];
  const intro=make('p','Select a row for the result, or “Command” for the intended action.','ladderIntro');
  const head=make('div',undefined,'ladderHead');head.append(make('span','Wall / commanded action'),make('span','Controller'),make('span','Attachment'));
  const ladder=make('div',undefined,'attemptLadder');ladder.append(head);let stage='';
  for(const [t,group,title] of definitions){const r=recordAt('robot',t);if(!r)throw Error('Missing alignment robot call '+t);
    if(group!==stage){const h=make('p',group,'ladderStage');ladder.append(h);stage=group;}
    const tool=TRACE.records.filter(e=>e.kind==='tool'&&e.t<=r.t&&r.t-e.t<1&&e.raw.arguments?.command?.includes('s.act')).at(-1);
    const state=TRACE.states.filter(s=>s.t<=r.end).at(-1),reason=r.raw.result?.act_result?.reason||'unknown';
    const row=make('div',undefined,'attemptRow'),button=make('button',undefined,'diagramEvent attemptEvent');button.dataset.source=r.id;button.setAttribute('aria-pressed','false');button.append(make('small',time(r.t)),make('span',title),make('span',reason,`controllerResult ${reason}`));button.addEventListener('click',()=>selectSource(r));
    const outcome=make('button',state.goal?'Satisfied':'Not satisfied',`attachmentResult ${state.goal?'satisfied':''}`),sr=stateRecord(state.step);byId.set(sr.id,sr);outcome.dataset.source=sr.id;outcome.setAttribute('aria-label',`Evaluator after ${title}: attachment ${state.goal?'satisfied':'not satisfied'}`);outcome.addEventListener('click',()=>selectSource(sr));row.append(button,outcome);
    if(tool){const command=make('button','Command ↗','commandLink');command.dataset.source=tool.id;command.addEventListener('click',()=>selectSource(tool));row.append(command);}ladder.append(row);
  }
  const footer=make('div',undefined,'ladderEvidence');footer.append(sourceButton('refine','Recompute from fresh RGB-D','Writes alignment_refined.json'),sourceButton('final','Send the revised joint target','Reads alignment_refined.json'));
  el('diagramBody').append(intro,ladder,footer);
  el('diagramCaption').textContent='“Reached” refers to the robot target. Attachment is measured separately by the evaluator.';
}
function setupStory() {
  setupClockInspection();
  document.querySelectorAll('.overviewEvent').forEach(a=>a.addEventListener('click',e=>{e.preventDefault();chooseMoment(moments.findIndex(m=>m.id===a.dataset.moment));el('moments').scrollIntoView({block:'start',behavior:'smooth'});}));
  for(const [kind,label] of [['model','Model'],['tool','Tools'],['robot','Robot'],['context','Context']]){const b=make('button',label,`filterChip ${kind}`);b.setAttribute('aria-pressed','true');b.addEventListener('click',()=>{traceKinds.has(kind)?traceKinds.delete(kind):traceKinds.add(kind);b.setAttribute('aria-pressed',String(traceKinds.has(kind)));renderRecordList();});el('traceFilters').append(b);}
  el('traceSearch').addEventListener('input',renderRecordList);buildStateLanes();renderRecordList();
}
