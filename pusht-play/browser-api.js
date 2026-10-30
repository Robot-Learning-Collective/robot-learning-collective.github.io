import init, { Sim, randomInit, world, controlHz, successThreshold, productScale, agentRadius, goalPose } from './assets/pusht_wasm.js';

export async function createPlayer() {
  async function fetchAsset(name, format) {
    const response = await fetch(new URL('./assets/' + name, import.meta.url));
    if (!response.ok) throw Error('Could not load the model. Reload to try again.');
    return response[format]();
  }
  const [, config, weights] = await Promise.all([
    init(), fetchAsset('policy.json', 'json'), fetchAsset('policy.bin', 'arrayBuffer')
  ]);
  const worker = new Worker(new URL('./policy-worker.js', import.meta.url), { type: 'module' });
  const requests = new Map();
  let requestId = 0, workerError = null;
  worker.onmessage = ({ data }) => {
    const pending = requests.get(data.id);
    if (!pending) return;
    requests.delete(data.id);
    clearTimeout(pending.timer);
    data.error ? pending.reject(Error(data.error)) : pending.resolve(data.actions);
  };
  worker.onerror = () => {
    workerError = Error('Model worker failed. Reload to try again.');
    for (const pending of requests.values()) { clearTimeout(pending.timer); pending.reject(workerError); }
    requests.clear();
  };
  function callWorker(data, transfer = []) {
    if (workerError) return Promise.reject(workerError);
    return new Promise((resolve, reject) => {
      const id = ++requestId;
      const timer = setTimeout(() => {
        requests.delete(id);
        reject(Error('Model inference timed out. Reload to try again.'));
      }, 30000);
      requests.set(id, { resolve, reject, timer });
      worker.postMessage({ ...data, id }, transfer);
    });
  }
  await callWorker({ type: 'load', config, buffer: weights }, [weights]);
  const sim = new Sim();
  let episode = 0, steps = 0, actions = [], action = null, done = false, success = false;
  const constants = { world: world(), control_hz: controlHz(), scale: productScale(),
    agent_radius: agentRadius(), goal: Array.from(goalPose()),
    outline: [[-60,-45],[60,-45],[60,-15],[15,-15],[15,75],[-15,75],[-15,-15],[-60,-15]] };

  function snapshot() {
    const red = Array.from(sim.helperPos());
    return { ...constants, episode, steps, done, success, action,
      state: [...sim.agentPos(), ...sim.blockPose()],
      coverage: sim.coverage(), human_pos: red.length ? red : null, plan: actions };
  }
  function reset() {
    sim.reset(...randomInit(crypto.getRandomValues(new Uint32Array(1))[0]));
    episode++; steps = 0; actions = []; action = null; done = success = false;
    return snapshot();
  }
  function join() {
    if (done) throw Error('Reset to start another episode.');
    if (sim.helperPos().length) return snapshot();
    const state = snapshot().state;
    const corners = [[40,40],[472,40],[40,472],[472,472]];
    const distance = p => Math.min(Math.hypot(p[0]-state[0],p[1]-state[1]), Math.hypot(p[0]-state[2],p[1]-state[3]));
    corners.sort((a,b) => distance(b)-distance(a));
    for (const p of corners) {
      try { sim.addHelper(...p); return snapshot(); } catch { /* Try another clear corner. */ }
    }
    throw Error('No clear place for the red robot. Try resetting.');
  }
  async function step(target) {
    if (done) return snapshot();
    if (!actions.length) {
      const predicted = await callWorker({ type: 'infer', state: snapshot().state });
      if (!predicted.every(Number.isFinite)) throw Error('Model produced an invalid action.');
      for (let i=0; i<config.chunk; i++) actions.push([predicted[2*i],predicted[2*i+1]]);
    }
    sim.setHelperTarget(target?.[0] ?? 0, target?.[1] ?? 0, !!target);
    action = actions.shift();
    sim.step(...action);
    steps++;
    success = sim.coverage() >= successThreshold();
    done = success || steps >= 300;
    return snapshot();
  }
  reset();
  return {
    async request(path, data = {}) {
      if (path === '/api/state') return snapshot();
      if (path === '/api/reset') return reset();
      if (data.episode !== episode) throw Error('Episode changed. Reset to continue.');
      if (path === '/api/join') return join();
      if (path === '/api/step') return step(data.human_target);
      throw Error('Unknown player action.');
    }
  };
}
