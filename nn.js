// nn.js — Minimal binary-classification neural network.
// Usable in both the browser (via <script src>) and Node.js (require/import).
//
// ── Architecture ──────────────────────────────────────────────────────────────
//
//   Input layer  →  hidden layer(s)  →  output layer (single sigmoid neuron)
//
//   sizes = [2, 32, 32, 1]  means:  2 inputs, two hidden layers of 32, 1 output.
//
// ── Forward pass (layer by layer) ────────────────────────────────────────────
//
//   For each hidden layer l:
//     z[l][j] = Σ_i  W[l][j][i] · a[l][i]  +  b[l][j]   (pre-activation)
//     a[l+1][j] = activation(z[l][j])                     (tanh or ReLU)
//
//   For the output layer:
//     z_out = W · a_last + b
//     p     = sigmoid(z_out)    →  probability in (0, 1)
//
// ── Loss — binary cross-entropy ──────────────────────────────────────────────
//
//   L = −y·log(p) − (1−y)·log(1−p)
//
//   This penalises confident wrong answers heavily (log → −∞ when p→0, y=1).
//
// ── Backpropagation ───────────────────────────────────────────────────────────
//
//   We want dL/dW and dL/db for every layer so we can do gradient descent.
//   We use the chain rule, propagating an error signal ("delta") backwards.
//
//   Output layer — a lucky cancellation:
//     dL/dp     = −y/p + (1−y)/(1−p)
//     dp/dz_out = p(1−p)                 (sigmoid derivative)
//     delta_out = dL/dz_out = (dL/dp)·(dp/dz_out)
//               = [−y/p + (1−y)/(1−p)] · p(1−p)
//               = p − y                  (the two pieces cancel cleanly)
//
//   Hidden layer l  (backwards from delta at layer l+1):
//     dL/da[l][i]    = Σ_j  W[l][j][i] · delta[l+1][j]   (chain through weights)
//     delta[l][i]    = dL/da[l][i] · activation′(a[l+1][i])
//                                                           (chain through activation)
//     Note: activation′ is expressed in terms of the activation VALUE, not z,
//     because tanh′(z) = 1 − tanh(z)² and relu′(z) = 1 if z > 0, so we don't
//     need to store z separately — only the stored activation is needed.
//
//   Weight and bias gradients (once we have delta for a layer):
//     dL/dW[l][j][i] = delta[j] · a_prev[i]
//     dL/db[l][j]    = delta[j]
//
// ─────────────────────────────────────────────────────────────────────────────

const sigmoid = x => 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, x))));

// Activation functions for hidden layers.
// Each entry has:
//   fn(z)        — the activation applied to the pre-activation z
//   derivative(a) — activation′ expressed as a function of the *activation value* a,
//                   so we can reuse the stored activation without re-computing z.
//   initScale(nin) — weight-init scale matched to the activation (avoids vanishing/exploding)
const ACTIVATIONS = {
  tanh: {
    fn:         z => Math.tanh(z),
    derivative: a => 1 - a * a,          // tanh′(z) = 1 − tanh(z)²
    initScale:  nin => Math.sqrt(1.0 / nin),  // Xavier / Glorot
  },
  relu: {
    fn:         z => Math.max(0, z),
    derivative: a => a > 0 ? 1 : 0,     // relu′(z) = 1 if z > 0, 0 otherwise
    initScale:  nin => Math.sqrt(2.0 / nin),  // He / Kaiming
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
    this.actFn      = act.fn;          // z → a
    this.actDeriv   = act.derivative;  // a → da/dz  (from activation value, not z)
    this.actName    = activation;

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

    // Pre-allocated activation buffers — one Float64Array per layer, reused every forward pass.
    // activations[0] holds the input; activations[L] holds the output probability.
    this.activations = sizes.map(n => new Float64Array(n));
  }

  // Forward pass.
  // x must be array-like with length == sizes[0].
  // Returns the scalar output probability in [0, 1].
  forward(x) {
    this.activations[0].set(x);

    for (let l = 0; l < this.L; l++) {
      const aPrev  = this.activations[l];
      const aCur   = this.activations[l + 1];
      const Wl     = this.W[l];
      const bl     = this.b[l];
      const nout   = aCur.length;
      const nin    = aPrev.length;
      const isLast = l === this.L - 1;

      for (let j = 0; j < nout; j++) {
        let z = bl[j];
        const Wlj = Wl[j];
        for (let i = 0; i < nin; i++) z += Wlj[i] * aPrev[i];
        // Hidden layers use tanh/ReLU; the output layer uses sigmoid to produce a probability.
        aCur[j] = isLast ? sigmoid(z) : this.actFn(z);
      }
    }

    return this.activations[this.L][0];
  }

  // Compute analytical gradients for a single sample.
  // forward() must have been called immediately before (activations must be fresh).
  // Returns { dW, db } — arrays of the same shape as this.W / this.b.
  // Does NOT modify weights.
  gradients(y) {
    const L   = this.L;
    const out = this.activations[L][0];

    // delta[j] = dL/dz[j] for the current layer being processed.
    // At the output layer, this is (p − y) — see the derivation at the top of the file.
    let delta = new Float64Array([out - y]);

    const dW = this.W.map(Wl => Wl.map(row => new Float64Array(row.length)));
    const db = this.b.map(bl => new Float64Array(bl.length));

    for (let l = L - 1; l >= 0; l--) {
      const aPrev = this.activations[l];  // activations feeding into layer l
      const Wl    = this.W[l];
      const nout  = delta.length;
      const nin   = aPrev.length;

      // ── Step 1: compute delta for the previous layer before we use delta here.
      //    (We read Wl to propagate delta backwards; the weights are unchanged.)
      let prevDelta = null;
      if (l > 0) {
        prevDelta = new Float64Array(nin);
        for (let i = 0; i < nin; i++) {
          let s = 0;
          for (let j = 0; j < nout; j++) s += Wl[j][i] * delta[j];
          // Chain through the activation derivative (expressed from the stored activation value)
          prevDelta[i] = s * this.actDeriv(aPrev[i]);
        }
      }

      // ── Step 2: accumulate gradients for this layer's weights and biases.
      //    dL/dW[l][j][i] = delta[j] · aPrev[i]
      //    dL/db[l][j]    = delta[j]
      for (let j = 0; j < nout; j++) {
        const dj = delta[j];
        for (let i = 0; i < nin; i++) dW[l][j][i] += dj * aPrev[i];
        db[l][j] += dj;
      }

      if (prevDelta) delta = prevDelta;
    }

    return { dW, db };
  }

  // Apply a gradient update (plain SGD): subtract lr * gradient from every parameter.
  applyGradients(dW, db, lr) {
    for (let l = 0; l < this.L; l++) {
      for (let j = 0; j < this.W[l].length; j++) {
        const Wlj = this.W[l][j];
        for (let i = 0; i < Wlj.length; i++) Wlj[i] -= lr * dW[l][j][i];
        this.b[l][j] -= lr * db[l][j];
      }
    }
  }

  // Mini-batch gradient descent step.
  // xs / ys: arrays of length B (the batch size).
  // Accumulates gradients over all B samples, then applies the *mean* gradient once.
  // Taking the mean rather than the sum makes the effective learning rate independent
  // of batch size, so the same lr works whether B = 1 (plain SGD) or B = 128.
  // Returns the mean cross-entropy loss over the batch.
  batchStep(xs, ys, lr) {
    const B = xs.length;

    // Gradient accumulators — same shape as W and b, initialised to zero.
    const dWacc = this.W.map(Wl => Wl.map(row => new Float64Array(row.length)));
    const dbacc = this.b.map(bl => new Float64Array(bl.length));
    let totalLoss = 0;

    for (let k = 0; k < B; k++) {
      const pred = this.forward(xs[k]);
      const y    = ys[k];
      totalLoss += -y * Math.log(pred + 1e-7) - (1 - y) * Math.log(1 - pred + 1e-7);

      const { dW, db } = this.gradients(y);

      // Add this sample's gradients to the accumulators.
      for (let l = 0; l < this.L; l++) {
        for (let j = 0; j < dW[l].length; j++) {
          for (let i = 0; i < dW[l][j].length; i++) dWacc[l][j][i] += dW[l][j][i];
          dbacc[l][j] += db[l][j];
        }
      }
    }

    // Dividing lr by B is equivalent to applying the mean gradient (sum/B) at rate lr.
    this.applyGradients(dWacc, dbacc, lr / B);
    return totalLoss / B;
  }

  // Convenience wrapper: one SGD step for a single sample (batch size = 1).
  // Returns the cross-entropy loss for this sample.
  step(x, y, lr) {
    return this.batchStep([x], [y], lr);
  }
}

// Export for Node.js
if (typeof module !== 'undefined') module.exports = { NN, sigmoid };
