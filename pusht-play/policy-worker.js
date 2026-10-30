import { StatePolicy, gaussianNoise } from './policy.js';

let policy;
self.onmessage = event => {
  const { id, type, config, buffer, state } = event.data;
  try {
    if (type === 'load') {
      policy = new StatePolicy(config, buffer);
      self.postMessage({ id });
    } else {
      const actions = policy.inference(state, gaussianNoise(policy.config.horizon * 2));
      self.postMessage({ id, actions }, [actions.buffer]);
    }
  } catch (error) {
    self.postMessage({ id, error: error.message });
  }
};
