"use strict";

const $ = id => document.getElementById(id);
const embedded = new URLSearchParams(location.search).get("embed") === "1";
if (embedded) {
  document.body.classList.add("embed");
  new ResizeObserver(() => {
    window.parent.postMessage(
      { type: "episode-viewer-height", height: Math.ceil(document.body.getBoundingClientRect().height) },
      location.origin
    );
  }).observe(document.body);
}
const video = $("video");
const speeds = [1, 2, 4, 8];
let manifest = [];
let episode = null;
let currentId = null;
let speed = 1;
let pendingSeek = null;

function time(seconds) {
  const n = Math.max(0, Math.floor(seconds || 0));
  return `${Math.floor(n / 60)}:${String(n % 60).padStart(2, "0")}`;
}

function qColor(value) {
  if (value == null) return "var(--dim)";
  const q = Math.min(1, Math.max(0, Number(value)));
  const hue = q <= 0.5 ? q * 84 : 42 + (q - 0.5) * 206;
  return `hsl(${hue} 75% 66%)`;
}

function frame() {
  return episode ? Math.min(episode.steps - 1, Math.max(0, Math.floor(video.currentTime * episode.fps))) : 0;
}

function seek(target) {
  if (!episode) return;
  const seconds = Math.max(0, Math.min(target, episode.steps - 1)) / episode.fps;
  if (!video.readyState) pendingSeek = seconds;
  else video.currentTime = seconds;
}

function addSegment(parent, className, start, end, label) {
  const element = document.createElement("span");
  element.className = className;
  element.style.left = `${100 * start / episode.steps}%`;
  if (end != null) element.style.width = `${100 * Math.max(0, end - start) / episode.steps}%`;
  if (label) element.title = label;
  parent.append(element);
  return element;
}

function renderNavigation() {
  const list = $("instance-list");
  list.replaceChildren();
  for (const item of manifest.filter(item => item.task === episode.task)) {
    const button = document.createElement("button");
    button.className = "instance";
    button.type = "button";
    button.setAttribute("aria-current", String(item.id === currentId));
    const name = document.createElement("span");
    name.textContent = `Instance ${item.instance}`;
    const q = document.createElement("span");
    q.className = "q";
    q.textContent = `Q ${item.q == null ? "—" : Number(item.q).toFixed(2)}`;
    q.style.color = qColor(item.q);
    button.append(name, q);
    button.addEventListener("click", () => select(item.id).catch(showError));
    list.append(button);
  }
}

function renderTimeline() {
  const holdTrack = $("hold-track");
  holdTrack.replaceChildren();
  $("goal-track").replaceChildren();
  const boundaries = [...new Set(episode.holds.flatMap(hold => [hold.start, hold.end]))].sort((a, b) => a - b);
  for (let i = 0; i < boundaries.length - 1; i++) {
    const start = boundaries[i];
    const end = boundaries[i + 1];
    const active = episode.holds.filter(hold => hold.start <= start && hold.end > start);
    if (!active.length) continue;
    const arms = new Set(active.map(hold => hold.arm));
    const kind = arms.size > 1 ? "both" : active[0].arm;
    const label = active.map(hold => `${hold.arm} arm holding ${hold.object}`).join(" · ");
    const segment = addSegment(holdTrack, `hold-segment ${kind}`, start, end);
    segment.dataset.label = label;
    segment.tabIndex = 0;
    segment.setAttribute("aria-label", label);
    if (start / episode.steps < 0.15) segment.classList.add("near-start");
    if (end / episode.steps > 0.85) segment.classList.add("near-end");
  }
  for (const event of episode.events.filter(event => event.type.startsWith("goal-"))) {
    addSegment($("goal-track"), `goal-mark ${event.type === "goal-off" ? "off" : ""}`, event.frame, null, event.label);
  }
  const maxStage = Math.max(1, ...episode.stages.map(item => item.value));
  const y = value => 34 - Math.max(0, Math.min(1, value / maxStage)) * 27;
  const stages = episode.stages;
  let path = stages.length ? `M 0 ${y(stages[0].value)}` : "";
  for (const stage of stages.slice(1)) {
    const x = 1000 * stage.frame / episode.steps;
    path += ` H ${x} V ${y(stage.value)}`;
  }
  if (path) path += " H 1000";
  $("stage-line").setAttribute("d", path);
}

function update() {
  if (!episode) return;
  const at = frame();
  $("time").textContent = `${time(video.currentTime)} / ${time(episode.steps / episode.fps)}`;
  $("playhead").style.left = `${100 * at / episode.steps}%`;
  $("stage-track").style.setProperty("--progress", `${100 * at / episode.steps}%`);
  $("timeline").setAttribute("aria-valuemax", String(episode.steps - 1));
  $("timeline").setAttribute("aria-valuenow", String(at));
  const stage = episode.stages.findLast(item => item.frame <= at);
  const maxStage = Math.max(0, ...episode.stages.map(item => item.value));
  $("stage-readout").textContent = `stage ${stage && stage.value >= 0 ? stage.value : "?"} / ${maxStage}`;
  const state = episode.goals.map(() => false);
  for (const event of episode.events) {
    if (event.frame > at) break;
    if (event.goal_index != null && event.goal_index < state.length) state[event.goal_index] = event.type === "goal-on";
  }
  [...$("goal-list").children].forEach((node, index) => {
    node.classList.toggle("met", state[index]);
    node.setAttribute("aria-label", `Goal ${index + 1}: ${node.dataset.goal}; ${state[index] ? "satisfied" : "not satisfied"}`);
  });
}

async function select(id) {
  const item = manifest.find(row => row.id === id);
  if (!item) throw new Error("Episode is absent from the public manifest");
  const response = await fetch(item.detail);
  if (!response.ok) throw new Error(`Episode data could not be loaded (${response.status})`);
  const data = await response.json();
  if (data.id !== id || !data.video_url || !data.fps || !data.steps) throw new Error("Incomplete episode data");
  video.pause();
  pendingSeek = null;
  currentId = id;
  episode = data;
  $("error").hidden = true;
  const url = new URL(location.href);
  url.searchParams.set("episode", id);
  history.replaceState(null, "", url);
  document.title = `${data.task.replaceAll("_", " ")} · Instance ${data.instance} · BEHAVIOR-1K`;
  $("task-name").textContent = data.task.replaceAll("_", " ");
  const q = document.createElement("span");
  q.className = "q";
  q.style.color = qColor(data.q);
  q.textContent = `Q ${data.q == null ? "—" : Number(data.q).toFixed(2)}`;
  $("episode-name").replaceChildren(`Instance ${data.instance} · `, q);
  video.src = data.video_url;
  video.playbackRate = speed;
  video.play().catch(() => {});
  $("goal-list").replaceChildren(...data.goals.map(goal => {
    const li = document.createElement("li");
    li.tabIndex = 0;
    li.dataset.goal = goal;
    return li;
  }));
  $("reason-section").hidden = !data.reason;
  $("reason").textContent = data.reason || "";
  renderNavigation();
  renderTimeline();
  update();
}

function showError(error) {
  $("error").hidden = false;
  $("error").textContent = error.message || String(error);
}

function setupSpeed() {
  try {
    const stored = Number(localStorage.getItem("behavior-public-playback-speed"));
    if (speeds.includes(stored)) speed = stored;
  } catch (_) { /* Storage may be unavailable. */ }
  for (const value of speeds) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = `${value}×`;
    button.setAttribute("aria-pressed", String(value === speed));
    button.addEventListener("click", () => {
      speed = value;
      video.playbackRate = value;
      try { localStorage.setItem("behavior-public-playback-speed", String(value)); } catch (_) {}
      for (const node of $("speeds").children) node.setAttribute("aria-pressed", String(node === button));
    });
    $("speeds").append(button);
  }
  video.playbackRate = speed;
  video.addEventListener("loadedmetadata", () => {
    video.playbackRate = speed;
    if (pendingSeek != null) { video.currentTime = pendingSeek; pendingSeek = null; }
    update();
  });
}

function setupSeek(id) {
  $(id).addEventListener("click", event => {
    const rect = event.currentTarget.getBoundingClientRect();
    seek(Math.round(episode.steps * (event.clientX - rect.left) / rect.width));
  });
}

setupSpeed();
setupSeek("timeline");
setupSeek("stage-track");
$("timeline").addEventListener("keydown", event => {
  if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
    event.preventDefault();
    seek(frame() + (event.key === "ArrowRight" ? 1 : -1) * episode.fps);
  }
});
video.addEventListener("timeupdate", update);
video.addEventListener("seeked", update);
video.addEventListener("error", () => { if (episode) showError(new Error("Video could not be loaded")); });

fetch("./data/manifest.json")
  .then(response => { if (!response.ok) throw new Error("Public manifest could not be loaded"); return response.json(); })
  .then(data => {
    manifest = data.episodes || [];
    if (!manifest.length) throw new Error("Public episode data has not been exported yet");
    return select(new URLSearchParams(location.search).get("episode") || manifest[0].id);
  })
  .catch(showError);
