'use strict';
let inspectedWall=903.07154;
function stateAtWall(t){let lo=0,hi=TRACE.states.length-1;while(lo<hi){const mid=Math.ceil((lo+hi)/2);if(TRACE.states[mid].t<=t)lo=mid;else hi=mid-1;}return TRACE.states[lo];}
function setClockInspection(t,frame=stateAtWall(t).frame){
  inspectedWall=Math.max(0,Math.min(TRACE.duration,t));
  const x=80+inspectedWall/TRACE.duration*860,y=275-frame/2788*180;
  const cursor=el('overviewCursor');cursor.setAttribute('cx',x);cursor.setAttribute('cy',y);
  for(const key of ['x1','x2'])el('overviewCrossX').setAttribute(key,x);
  for(const key of ['y1','y2'])el('overviewCrossY').setAttribute(key,y);
  const text=`Wall ${time(inspectedWall)} · video ${time(frame/30)} · frame ${frame}`;
  el('overviewReadout').textContent=text;el('clockInspector').setAttribute('aria-valuenow',inspectedWall);el('clockInspector').setAttribute('aria-valuetext',text);
}
function setupClockInspection(){
  const svg=el('clockChart');
  svg.addEventListener('pointermove',e=>{const p=svg.createSVGPoint();p.x=e.clientX;p.y=e.clientY;const pt=p.matrixTransform(svg.getScreenCTM().inverse());if(pt.x>=80&&pt.x<=940&&pt.y>=85&&pt.y<=280)setClockInspection((pt.x-80)/860*TRACE.duration);});
  svg.addEventListener('pointerleave',()=>setClockInspection(selected.t,selected.frame));
  el('clockInspector').addEventListener('keydown',e=>{let t=inspectedWall;if(e.key==='ArrowLeft')t-=e.shiftKey?10:1;else if(e.key==='ArrowRight')t+=e.shiftKey?10:1;else if(e.key==='Home')t=0;else if(e.key==='End')t=TRACE.duration;else return;e.preventDefault();setClockInspection(Math.max(0,Math.min(TRACE.duration,t)));});
}
function renderOrientationZoom(){
  el('diagramTitle').textContent='Orientation correction · clock detail';
  const start=900,end=950,low=34,high=40,svg=chart('Video time against wall time, 15:00–15:50',330),x=t=>80+(t-start)/(end-start)*860,y=f=>255-(f/30-low)/(high-low)*165;
  svg.append(svgNode('rect',{x:x(sources.orientation.t),y:85,width:x(moments[0].end)-x(sources.orientation.t),height:170,class:'segmentBand'}));
  for(const seconds of [34,36,38,40]){const yy=y(seconds*30);line(svg,80,yy,940,yy,'grid');label(svg,65,yy+4,`${seconds} s`,'chartText','end');}
  for(const t of [900,910,920,930,940,950])label(svg,x(t),289,time(t).slice(0,5),'chartText',t===900?'start':t===950?'end':'middle');
  label(svg,80,30,'VIDEO TIME · SAME EVALUATOR CURVE','chartText');
  let d=`M80,${y(stateAtWall(start).frame)}`;for(const s of TRACE.states.filter(s=>s.t>start&&s.t<=end))d+=`H${x(s.t)}V${y(s.frame)}`;d+=`H940`;
  svg.append(svgNode('path',{d,class:'curve'}));
  const interval=intervals[1065];line(svg,x(interval.start),y(1065),x(interval.end),y(1065),'plateau');
  [['orientation','Orientation report'],['ik','IK · first set'],['ik2','IK · second set'],['correct','Motion command']].forEach(([key,text],i)=>{
    const r=sources[key],xx=x(r.t),yy=y(r.frame);line(svg,xx,yy-6,xx,yy-35,'segmentLeader');
    eventPoint(svg,key,xx,yy,text,`${time(r.t)} wall`,{anchor:i===0?'start':i===3?'end':'middle',dy:-58});
  });
  label(svg,940,317,'WALL TIME','chartText','end');
  el('diagramCaption').textContent=`Zoom of the timing overview, 15:00–15:50 wall time. Frame 1065 (35.5 s of video) persists for ${(interval.end-interval.start).toFixed(1)} s; the four record starts fall on this flat stretch.`;
}
function renderObservationComparison(before,after){
  const root=el('observations');root.replaceChildren();
  if(!before||!after){root.append(observationFigure(before,'Before the segment'),observationFigure(after,'Last observation in the segment'));return;}
  const stage=make('div',undefined,'comparisonStage'),a=make('img'),b=make('img');
  a.src=mediaURL(after.images[0]);a.alt=`After: tool observation returned at ${time(after.end)} wall`;b.src=mediaURL(before.images[0]);b.alt=`Before: tool observation returned at ${time(before.end)} wall`;b.className='comparisonBefore';
  stage.style.setProperty('--split','50%');stage.append(a,b,make('span','Before','compareLabel before'),make('span','After','compareLabel after'),make('div',undefined,'comparisonDivider'));
  const slider=make('input');slider.type='range';slider.min='0';slider.max='100';slider.value='50';slider.id='comparisonSplit';slider.setAttribute('aria-label','Before and after image comparison');slider.setAttribute('aria-valuetext','50 percent before, 50 percent after');
  const update=()=>{stage.style.setProperty('--split',`${slider.value}%`);slider.setAttribute('aria-valuetext',`${slider.value} percent before, ${100-Number(slider.value)} percent after`);};
  slider.addEventListener('input',update);let dragging=false;const move=e=>{const box=stage.getBoundingClientRect();slider.value=String(Math.round(Math.max(0,Math.min(100,(e.clientX-box.left)/box.width*100))));update();};
  stage.addEventListener('pointerdown',e=>{dragging=true;stage.setPointerCapture(e.pointerId);move(e);});stage.addEventListener('pointermove',e=>{if(dragging)move(e);});stage.addEventListener('pointerup',()=>dragging=false);stage.addEventListener('lostpointercapture',()=>dragging=false);
  const controls=make('div',undefined,'comparisonControls');controls.append(make('label','Drag to compare · original camera layout'));controls.firstChild.htmlFor=slider.id;controls.append(slider);
  const captions=make('div',undefined,'comparisonCaptions');for(const [r,name] of [[before,'Before'],[after,'After']]){const link=make('a',`${name} · returned ${time(r.end)} wall · open original ↗`);link.href=mediaURL(r.images[0]);link.target='_blank';link.rel='noreferrer';captions.append(link);}
  root.append(stage,controls,captions);
}
function renderMediaSegment(){
  const m=moments[active];el('videoSeek').style.setProperty('--segment-start',`${m.startFrame/2788*100}%`);el('videoSeek').style.setProperty('--segment-end',`${m.endFrame/2788*100}%`);el('videoBands').replaceChildren();
  moments.forEach((segment,index)=>{const band=make('button',undefined,`videoBand ${index===active?'active':''}`);band.style.left=`${segment.startFrame/2788*100}%`;band.style.width=`${(segment.endFrame-segment.startFrame)/2788*100}%`;band.title=`0${index+1} ${segment.label} · ${time(segment.startFrame/30)}–${time(segment.endFrame/30)}`;band.setAttribute('aria-label',band.title);band.setAttribute('aria-pressed',String(index===active));band.addEventListener('click',()=>chooseMoment(index));el('videoBands').append(band);});
  el('scrubberCaption').textContent=`Highlighted: 0${active+1} ${m.label} · ${time(m.startFrame/30)}–${time(m.endFrame/30)}`;
  const quote=el('segmentQuote');quote.hidden=active!==0;quote.replaceChildren();
  if(active===0){const body=sources.orientation.excerpt.replace(/^\*\*.*?\*\*\s*/,'');const first="I've flipped the object upside down!",last='I can adjust it using inverse kinematics.';
    if(!body.includes(first)||!body.includes(last))throw Error('Summary excerpt does not match source');
    quote.append(make('p',`“${first} … ${last}”`));const cite=make('button','Model summary · excerpt · 15:03.1 ↗');cite.addEventListener('click',()=>{selectSource(sources.orientation);el('selectedSource').scrollIntoView({block:'center',behavior:'smooth'});});quote.append(cite);
  }
}
function setupMediaControls(){
  const seek=el('videoSeek');seek.max='2788';
  seek.addEventListener('input',()=>{if(!ready)return;video.pause();playingMoment=false;video.currentTime=Number(seek.value)/30;refreshClock();});
  el('togglePlayback').addEventListener('click',async()=>{if(!ready)return;if(video.paused){playingMoment=false;try{await video.play();}catch{}}else video.pause();});
  el('videoFullscreen').addEventListener('click',()=>video.requestFullscreen?.());
  const update=()=>{el('togglePlayback').textContent=video.paused?'▶ Play':'Ⅱ Pause';el('togglePlayback').setAttribute('aria-label',video.paused?'Play episode video':'Pause episode video');};
  video.addEventListener('play',update);video.addEventListener('pause',update);
}
function recordPreview(r){
  if(r.kind==='summary')return r.excerpt.replace(/^\*\*.*?\*\*\s*/,'').split('\n').find(s=>s.trim())||'Returned summary';
  if(r.kind==='tool'){
    if(r.raw.arguments?.path)return `Read ${r.raw.arguments.path}`;
    const command=r.raw.arguments?.command||r.raw.arguments?.code||'';
    const lines=command.split('\n').map(s=>s.trim()).filter(Boolean);
    return lines.find(s=>!/^python\d?\s|^import\s|^from\s|^PY$/.test(s))||lines[0]||r.title;
  }
  if(r.kind==='model')return r.raw.text?.trim().split('\n')[0]||`Response ${time(r.t)} · ${r.raw.usage?.output??'unknown'} output tokens · text not returned`;
  return r.title;
}
