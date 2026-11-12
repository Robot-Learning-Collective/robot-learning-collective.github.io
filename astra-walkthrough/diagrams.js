'use strict';

// Coordinates are computed from the source timestamps or evaluator frames.
// The file diagram alone is schematic: its arrows represent a recorded write/read.
function svgNode(tag, attrs={}, text) {
  const n=document.createElementNS('http://www.w3.org/2000/svg',tag);
  for(const [k,v] of Object.entries(attrs))n.setAttribute(k,v);
  if(text!==undefined)n.textContent=text;
  return n;
}
function chart(title, height=230) {
  const scroll=make('div',undefined,'chartScroll');
  scroll.tabIndex=0;scroll.setAttribute('aria-label',`${title}; scroll horizontally on small screens`);
  const svg=svgNode('svg',{viewBox:`0 0 1000 ${height}`,role:'group','aria-label':title,class:'episodeChart'});
  scroll.append(svg);el('diagramBody').append(scroll,make('p','Scroll figure horizontally →','chartScrollHint'));return svg;
}
function line(svg,x1,y1,x2,y2,cls='chartLine') {svg.append(svgNode('line',{x1,y1,x2,y2,class:cls}));}
function label(svg,x,y,text,cls='chartText',anchor='start') {svg.append(svgNode('text',{x,y,class:cls,'text-anchor':anchor},text));}
function eventPoint(svg,key,x,y,title,sub,{anchor='middle',dy=-28}={}) {
  const r=sources[key],g=svgNode('g',{class:'diagramEvent',role:'button',tabindex:0,'data-source':r.id,'aria-pressed':'false','aria-label':`${title}, ${sub}; inspect source`});
  const width=Math.max(title.length,sub.length)*8+12,left=anchor==='start'?x-8:anchor==='end'?x-width+8:x-width/2,top=Math.min(y-20,y+dy-17),bottom=Math.max(y+20,y+dy+22);
  g.append(svgNode('rect',{x:left,y:top,width,height:bottom-top,fill:'transparent','pointer-events':'all'}));
  // A generous invisible hit area also keeps keyboard focus visible.
  g.append(svgNode('circle',{cx:x,cy:y,r:19,class:'eventHit'}),svgNode('circle',{cx:x,cy:y,r:5,class:'eventDot'}));
  g.append(svgNode('text',{x,y:y+dy,'text-anchor':anchor,class:'eventLabel'},title),svgNode('text',{x,y:y+dy+17,'text-anchor':anchor,class:'chartText'},sub));
  g.addEventListener('click',()=>selectSource(r));
  g.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();selectSource(r);}});
  svg.append(g);
}
function sourceButton(key,title,detail) {
  const b=make('button',undefined,'diagramEvent fileEvent');b.dataset.source=sources[key].id;b.setAttribute('aria-pressed','false');
  b.append(make('strong',title),make('span',detail),make('small',`${time(sources[key].t)} wall · inspect ↗`));
  b.addEventListener('click',()=>selectSource(sources[key]));return b;
}
function renderEpisodeDiagram(index) {
  el('diagramBody').replaceChildren();
  const title=el('diagramTitle'),caption=el('diagramCaption');
  if(index===0) {
    renderOrientationZoom();
  } else if(index===1) {
    title.textContent='Grasp transfer';
    const m=moments[index],svg=chart(title.textContent,280),x=f=>125+(f-m.startFrame)/(m.endFrame-m.startFrame)*750;
    const states=TRACE.states.filter(s=>s.frame>=m.startFrame&&s.frame<=m.endFrame);
    [['left','Left hand',132],['right','Right hand',174]].forEach(([key,text,y])=>{
      label(svg,20,y+20,text);let start=states[0];
      const draw=(s,end)=>{if(end<=s.frame)return;svg.append(svgNode('rect',{x:x(s.frame),y,width:x(end)-x(s.frame),height:28,class:s[key]?'graspHeld':'graspEmpty'}));};
      for(let i=1;i<states.length;i++)if(states[i][key]!==start[key]){draw(start,states[i].frame);start=states[i];}
      draw(start,m.endFrame);
    });
    [['close','Close right',55,'start'],['both','Both hold',55,'end'],['release','Open left',55,'end'],['right','Right only',231,'start']].forEach(([key,text,y,anchor])=>{
      const xx=x(sources[key].frame);line(svg,xx,y<100?66:208,xx,y<100?202:220,'chartGuide');
      eventPoint(svg,key,xx,y,text,`f ${sources[key].frame}`,{anchor,dy:y>200?20:-28});
    });
    label(svg,125,112,`Video ${time(m.startFrame/30)}`,'chartText');label(svg,875,112,time(m.endFrame/30),'chartText','end');
    const legend=make('div',undefined,'graspLegend');legend.append(make('span','Holding poster','heldKey'),make('span','No grasp','emptyKey'),make('span','Evaluator measurements'));el('diagramBody').append(legend);
    const frames=sources.right.frame-sources.both.frame;
    caption.textContent=`Both hands hold for ${frames} frames (${(frames/30).toFixed(2)} video seconds; ${(sources.right.t-sources.both.t).toFixed(2)} wall seconds). Bars show evaluator grasp states on a video-frame axis.`;
  } else if(index===2) {
    renderAlignmentLadder();
  } else {
    title.textContent='Attachment before call closure';
    const svg=chart(title.textContent),start=sources.final.t,end=sources.robot.end,x=t=>90+(t-start)/(end-start)*820;
    svg.append(svgNode('rect',{x:x(sources.robot.t),y:112,width:910-x(sources.robot.t),height:26,class:'callSpan'}));
    label(svg,110,130,'Final robot call','spanLabel');
    const gx=x(sources.goal.t);line(svg,gx,90,gx,149,'goalGuide');
    line(svg,gx,171,910,171,'anchorLine');
    label(svg,(gx+910)/2,196,`${(end-sources.goal.t).toFixed(2)} s to closure`,'anchorLabel','middle');
    eventPoint(svg,'final',90,90,'Command issued',time(sources.final.t)+' wall',{anchor:'start'});
    eventPoint(svg,'goal',gx,90,'Goal satisfied',time(sources.goal.t)+' wall',{anchor:'end'});
    eventPoint(svg,'robot',910,90,'Call closes',time(end)+' wall',{anchor:'end'});
    caption.textContent='Wall-time axis. Evaluator success and the unknown robot acknowledgement are separate records.';
  }
}
function syncDiagramSource(r) {
  const hasEvent=[...el('diagramBody').querySelectorAll('[data-source]')].some(n=>n.dataset.source===r.id);
  el('diagramSelection').replaceChildren();
  if(!hasEvent)return;
  const prefix=r===sources.robot?`Call ends ${time(r.end)} wall. Record starts ${time(r.t)} at frame ${r.frame} · `:`Selected: ${time(r.t)} wall · frame ${r.frame} · `;
  const text=make('span',prefix),link=make('a','View source record ↓');link.href='#selectedSource';
  el('diagramSelection').append(text,link);
}
