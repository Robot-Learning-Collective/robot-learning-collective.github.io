// Exact StateFlowPolicy architecture; EMA FP32 weights from best.pt.
export class StatePolicy {
  constructor(config, buffer) {
    this.config = config;
    this.weights = {};
    for (const [name, spec] of Object.entries(config.tensors)) {
      this.weights[name] = new Float32Array(buffer, spec.offset * 4, spec.length);
    }
  }

  dense(input, name) {
    const w = this.weights[name + '.weight'], bias = this.weights[name + '.bias'];
    const out = new Float32Array(bias.length), width = input.length;
    for (let i = 0; i < out.length; i++) {
      let sum = 0;
      const base = i * width;
      for (let j = 0; j < width; j++) sum += w[base + j] * input[j];
      out[i] = sum + bias[i];
    }
    return out;
  }

  layerNorm(input, name) {
    let mean = 0, variance = 0;
    for (const x of input) mean += x;
    mean /= input.length;
    for (const x of input) variance += (x - mean) ** 2;
    const scale = 1 / Math.sqrt(variance / input.length + 1e-5);
    const w = this.weights[name + '.weight'], b = this.weights[name + '.bias'];
    return Float32Array.from(input, (x, i) => (x - mean) * scale * w[i] + b[i]);
  }

  forward(actions, t, state) {
    const input = new Float32Array(actions.length + 22);
    input.set(actions);
    let offset = actions.length;
    for (let i = 0; i < 4; i++) input[offset++] = Math.fround(state[i]) / 256 - 1;
    // Match the policy's wrapping of the fifth observation into [-pi, pi).
    const angle = Math.fround(state[4]);
    input[offset++] = (((angle + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI) / Math.PI;
    input[offset++] = t;
    const freq = this.weights.time_freq;
    for (let i = 0; i < 8; i++) input[offset + i] = Math.sin(Math.fround(Math.fround(t) * freq[i]));
    for (let i = 0; i < 8; i++) input[offset + 8 + i] = Math.cos(Math.fround(Math.fround(t) * freq[i]));
    let x = this.dense(input, 'input');
    for (let i = 0; i < this.config.layers; i++) {
      const prefix = `blocks.${i}`;
      const normalized = this.layerNorm(x, prefix + '.0');
      const hidden = this.dense(normalized, prefix + '.1');
      for (let j = 0; j < hidden.length; j++) hidden[j] /= 1 + Math.exp(-hidden[j]);
      const residual = this.dense(hidden, prefix + '.3');
      for (let j = 0; j < x.length; j++) x[j] += residual[j];
    }
    return this.dense(x, 'output');
  }

  inference(state, noise) {
    const x = new Float32Array(noise);
    const steps = this.config.denoise_steps;
    for (let i = 0; i < steps; i++) {
      const velocity = this.forward(x, i / steps, state);
      for (let j = 0; j < x.length; j++) x[j] += Math.fround(velocity[j] / steps);
    }
    return Float32Array.from(x, value => (Math.max(-1, Math.min(1, value)) + 1) * 256);
  }
}

export function gaussianNoise(length) {
  const noise = new Float32Array(length);
  for (let i = 0; i < length; i += 2) {
    const radius = Math.sqrt(-2 * Math.log(1 - Math.random()));
    const phase = 2 * Math.PI * Math.random();
    noise[i] = radius * Math.cos(phase);
    if (i + 1 < length) noise[i + 1] = radius * Math.sin(phase);
  }
  return noise;
}
