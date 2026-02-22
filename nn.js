// nn.js — Neural network
// Usable in both the browser (via <script src>) and Node.js (require/import).

const sigmoid = x => 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, x))));

// Activation functions for hidden layers.
// Each entry has:
//   fn(z)  — the activation applied to the pre-activation z
//   dact(a) — the derivative of the activation, expressed in terms of the
//              *activation value* a (not z), so we don't need to store z separately.
//   initScale(nin) — weight initialisation scale appropriate for this activation.
const ACTIVATIONS = {
  tanh: {
    fn:        z => Math.tanh(z),
    dact:      a => 1 - a * a,          // tanh′(z) = 1 − tanh(z)²
    initScale: nin => Math.sqrt(1.0 / nin),  // Xavier / Glorot
  },
  relu: {
    fn:        z => Math.max(0, z),
    dact:      a => a > 0 ? 1 : 0,     // relu′(z) = 1 if z>0, 0 otherwise
    initScale: nin => Math.sqrt(2.0 / nin),  // He / Kaiming
  },
};

class NN {
  // sizes:      array of layer widths, e.g. [2, 32, 32, 1]
  // activation: 'tanh' (default) or 'relu'
  constructor(sizes, activation = 'tanh') {
    this.sizes = sizes;
    this.L = sizes.length - 1;  // number of weight layers
    this.W = [];                 // W[l][j][i]: weight from neuron i in layer l to neuron j in layer l+1
    this.b = [];                 // b[l][j]:    bias of neuron j in layer l+1

    const act = ACTIVATIONS[activation] ?? ACTIVATIONS.tanh;
    this.actFn   = act.fn;    // z  → a
    this.dact    = act.dact;  // a  → da/dz  (derivative from activation value)
    this.actName = activation;

    for (let l = 0; l < this.L; l++) {
      const nin = sizes[l], nout = sizes[l + 1];
      const scale = act.initScale(nin);
      const Wl = [];
      for (let j = 0; j < nout; j++) {
        const row = new Float64Array(nin);
        for (let i = 0; i < nin; i++) row[i] = (Math.random() * 2 - 1) * scale;
        Wl.push(row);
      }
      this.W.push(Wl);
      this.b.push(new Float64Array(nout));
    }

    // Pre-allocated activation buffers — reused on every forward pass
    this._a = sizes.map(n => new Float64Array(n));
  }

  // Forward pass.
  // x must be array-like with length == sizes[0].
  // Returns the scalar output (probability in [0, 1]).
  forward(x) {
    this._a[0].set(x);

    for (let l = 0; l < this.L; l++) {
      const aPrev   = this._a[l];
      const aCur    = this._a[l + 1];
      const Wl      = this.W[l];
      const bl      = this.b[l];
      const nout    = aCur.length;
      const nin     = aPrev.length;
      const isLast  = l === this.L - 1;

      for (let j = 0; j < nout; j++) {
        let s = bl[j];
        const Wlj = Wl[j];
        for (let i = 0; i < nin; i++) s += Wlj[i] * aPrev[i];
        aCur[j] = isLast ? sigmoid(s) : this.actFn(s);
      }
    }

    return this._a[this.L][0];
  }

  // Compute analytical gradients for a single sample.
  // forward() must have been called immediately before.
  // Returns { dW, db } — arrays of the same shape as this.W / this.b.
  // Does NOT modify weights.
  gradients(y) {
    const L   = this.L;
    const out = this._a[L][0];

    // For cross-entropy loss + sigmoid output: dL/dz_out = output − y
    let delta = new Float64Array([out - y]);

    const dW = this.W.map(Wl => Wl.map(row => new Float64Array(row.length)));
    const db = this.b.map(bl => new Float64Array(bl.length));

    for (let l = L - 1; l >= 0; l--) {
      const aPrev = this._a[l];  // activation feeding into layer l  (= tanh output of layer l-1)
      const Wl    = this.W[l];
      const nout  = delta.length;
      const nin   = aPrev.length;

      // ── Step 1: propagate delta to the previous layer BEFORE touching weights
      //    Uses pre-update weights, which is what we want.
      let nextDelta = null;
      if (l > 0) {
        nextDelta = new Float64Array(nin);
        for (let i = 0; i < nin; i++) {
          let s = 0;
          for (let j = 0; j < nout; j++) s += Wl[j][i] * delta[j];
          nextDelta[i] = s * this.dact(aPrev[i]);
        }
      }

      // ── Step 2: accumulate gradients for this layer
      for (let j = 0; j < nout; j++) {
        const dj = delta[j];
        for (let i = 0; i < nin; i++) dW[l][j][i] += dj * aPrev[i];
        db[l][j] += dj;
      }

      if (nextDelta) delta = nextDelta;
    }

    return { dW, db };
  }

  // Apply a gradient update (plain SGD).
  applyGradients(dW, db, lr) {
    for (let l = 0; l < this.L; l++) {
      for (let j = 0; j < this.W[l].length; j++) {
        const Wlj = this.W[l][j];
        for (let i = 0; i < Wlj.length; i++) Wlj[i] -= lr * dW[l][j][i];
        this.b[l][j] -= lr * db[l][j];
      }
    }
  }

  // One SGD step for a single sample.
  // Returns the cross-entropy loss for this sample.
  step(x, y, lr) {
    const pred = this.forward(x);
    const loss = -y * Math.log(pred + 1e-7) - (1 - y) * Math.log(1 - pred + 1e-7);
    const { dW, db } = this.gradients(y);
    this.applyGradients(dW, db, lr);
    return loss;
  }

  // Mini-batch gradient descent step.
  // xs / ys: arrays of length batchSize.
  // Accumulates gradients over all samples, then applies the *mean* gradient once.
  // One batchStep update is much less noisy than batchSize individual SGD steps.
  // Returns the mean cross-entropy loss over the batch.
  batchStep(xs, ys, lr) {
    const B = xs.length;

    // Gradient accumulators — same shape as W and b
    const dWacc = this.W.map(Wl => Wl.map(row => new Float64Array(row.length)));
    const dbacc = this.b.map(bl => new Float64Array(bl.length));
    let totalLoss = 0;

    for (let k = 0; k < B; k++) {
      const pred = this.forward(xs[k]);
      const y    = ys[k];
      totalLoss += -y * Math.log(pred + 1e-7) - (1 - y) * Math.log(1 - pred + 1e-7);

      const { dW, db } = this.gradients(y);

      for (let l = 0; l < this.L; l++) {
        for (let j = 0; j < dW[l].length; j++) {
          for (let i = 0; i < dW[l][j].length; i++) dWacc[l][j][i] += dW[l][j][i];
          dbacc[l][j] += db[l][j];
        }
      }
    }

    // lr / B  →  apply the sum of gradients scaled to the mean
    this.applyGradients(dWacc, dbacc, lr / B);
    return totalLoss / B;
  }
}

// Export for Node.js
if (typeof module !== 'undefined') module.exports = { NN, sigmoid };
