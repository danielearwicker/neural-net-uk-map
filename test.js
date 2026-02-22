// test.js — run with:  node test.js
'use strict';
const { NN } = require('./nn.js');

// ─────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────
function crossEntropy(pred, y) {
  return -y * Math.log(pred + 1e-7) - (1 - y) * Math.log(1 - pred + 1e-7);
}

function pass(label) { console.log(`  ✓ ${label}`); }
function fail(label) { console.log(`  ✗ FAIL: ${label}`); process.exitCode = 1; }
function check(ok, label) { ok ? pass(label) : fail(label); }

// ─────────────────────────────────────────────────────────────────
// Test 1: Numerical gradient check — both activations
// ─────────────────────────────────────────────────────────────────
function testGradients() {
  console.log('\n[1] Gradient check (numerical vs analytical)');

  // Test with tanh first, then relu
  for (const act of ['tanh', 'relu']) {
    testGradientsFor(act);
  }
}

function testGradientsFor(act) {
  const net = new NN([2, 4, 4, 1], act);
  const x = new Float64Array([0.6, -0.4]);
  const y = 1;
  const eps = 1e-5;

  net.forward(x);
  const { dW, db } = net.gradients(y);

  let maxRelErr = 0;

  // Check every weight in every layer
  for (let l = 0; l < net.L; l++) {
    for (let j = 0; j < net.W[l].length; j++) {
      for (let i = 0; i < net.W[l][j].length; i++) {
        const orig = net.W[l][j][i];

        net.W[l][j][i] = orig + eps;
        const lp = crossEntropy(net.forward(x), y);

        net.W[l][j][i] = orig - eps;
        const lm = crossEntropy(net.forward(x), y);

        net.W[l][j][i] = orig;

        const numGrad = (lp - lm) / (2 * eps);
        const anaGrad = dW[l][j][i];
        const relErr  = Math.abs(numGrad - anaGrad) / (Math.abs(numGrad) + Math.abs(anaGrad) + 1e-10);
        if (relErr > maxRelErr) maxRelErr = relErr;
      }
    }
  }

  // Check every bias
  for (let l = 0; l < net.L; l++) {
    for (let j = 0; j < net.b[l].length; j++) {
      const orig = net.b[l][j];

      net.b[l][j] = orig + eps;
      const lp = crossEntropy(net.forward(x), y);

      net.b[l][j] = orig - eps;
      const lm = crossEntropy(net.forward(x), y);

      net.b[l][j] = orig;

      const numGrad = (lp - lm) / (2 * eps);
      const anaGrad = db[l][j];
      const relErr  = Math.abs(numGrad - anaGrad) / (Math.abs(numGrad) + Math.abs(anaGrad) + 1e-10);
      if (relErr > maxRelErr) maxRelErr = relErr;
    }
  }

  console.log(`  ${act}: max relative error: ${maxRelErr.toExponential(2)}`);
  check(maxRelErr < 1e-4, `${act}: all gradients within tolerance`);
}

// ─────────────────────────────────────────────────────────────────
// Test 2: Loss decreases on a fixed single sample
// ─────────────────────────────────────────────────────────────────
function testLossDecreases() {
  console.log('\n[2] Loss decreases on a single repeated sample');

  const net = new NN([2, 8, 1]);
  const x = new Float64Array([0.5, -0.3]);
  const y = 1;

  const loss0 = crossEntropy(net.forward(x), y);
  for (let i = 0; i < 200; i++) net.step(x, y, 0.1);
  const loss1 = crossEntropy(net.forward(x), y);

  console.log(`  Loss before: ${loss0.toFixed(4)},  after 200 steps: ${loss1.toFixed(4)}`);
  check(loss1 < loss0, 'Loss decreased');
  check(loss1 < 0.05,  'Loss is small (overfit single sample)');
}

// ─────────────────────────────────────────────────────────────────
// Test 3: Learn a linearly separable problem
// ─────────────────────────────────────────────────────────────────
function testLinearSep() {
  console.log('\n[3] Learn linearly separable: y = 1 if x0 > 0');

  const net = new NN([2, 8, 1]);
  const N = 500;
  const xs = Array.from({ length: N }, () => {
    const x0 = Math.random() * 2 - 1;
    const x1 = Math.random() * 2 - 1;
    return new Float64Array([x0, x1]);
  });
  const ys = xs.map(x => x[0] > 0 ? 1 : 0);

  for (let step = 0; step < 5000; step++) {
    const k = step % N;
    net.step(xs[k], ys[k], 0.05);
  }

  let correct = 0;
  for (let i = 0; i < N; i++) {
    if ((net.forward(xs[i]) > 0.5 ? 1 : 0) === ys[i]) correct++;
  }
  const acc = correct / N;
  console.log(`  Accuracy: ${(acc * 100).toFixed(1)}%`);
  check(acc > 0.95, 'Accuracy > 95%');
}

// ─────────────────────────────────────────────────────────────────
// Test 4: Learn a non-linear problem — unit circle, both activations
// ─────────────────────────────────────────────────────────────────
function testCircle() {
  console.log('\n[4] Learn non-linear: inside unit circle (tanh and relu)');

  const N = 1000;
  const xs = Array.from({ length: N }, () =>
    new Float64Array([Math.random() * 2 - 1, Math.random() * 2 - 1])
  );
  const ys = xs.map(x => (x[0] ** 2 + x[1] ** 2) < 0.7 ? 1 : 0);

  for (const act of ['tanh', 'relu']) {
    const net = new NN([2, 16, 16, 1], act);
    for (let step = 0; step < 20000; step++) {
      net.step(xs[step % N], ys[step % N], 0.05);
    }
    let correct = 0;
    for (let i = 0; i < N; i++) {
      if ((net.forward(xs[i]) > 0.5 ? 1 : 0) === ys[i]) correct++;
    }
    const acc = correct / N;
    console.log(`  ${act}: accuracy = ${(acc * 100).toFixed(1)}%`);
    check(acc > 0.90, `${act}: accuracy > 90%`);
  }
}

// ─────────────────────────────────────────────────────────────────
// Test 5: Mini-batch loss variance < single-sample loss variance
//
// This is a statistical fact (law of large numbers) independent of training:
// the mean of B losses has variance = (single-sample variance) / B.
// We verify it on a fixed, partially-trained network.
// ─────────────────────────────────────────────────────────────────
function testMiniBatchSmoother() {
  console.log('\n[5] Mini-batch loss variance is lower than single-sample loss variance');

  const BATCH = 32;
  const N = 500;
  const xs = Array.from({ length: N }, () =>
    new Float64Array([Math.random() * 2 - 1, Math.random() * 2 - 1])
  );
  const ys = xs.map(x => x[0] > 0 ? 1 : 0);

  // Train a single network to a reasonable state
  const net = new NN([2, 16, 1]);
  for (let s = 0; s < 3000; s++) net.step(xs[s % N], ys[s % N], 0.05);

  // Measure loss variance WITHOUT touching weights (evaluate only, no update)
  // Single-sample losses: compute cross-entropy for 200 individual random samples
  const singleLosses = [];
  for (let s = 0; s < 200; s++) {
    const k    = Math.floor(Math.random() * N);
    const pred = net.forward(xs[k]);
    const y    = ys[k];
    singleLosses.push(-y * Math.log(pred + 1e-7) - (1 - y) * Math.log(1 - pred + 1e-7));
  }

  // Batch-mean losses: average of BATCH samples each time, 200 times
  const batchLosses = [];
  for (let s = 0; s < 200; s++) {
    let sum = 0;
    for (let b = 0; b < BATCH; b++) {
      const k    = Math.floor(Math.random() * N);
      const pred = net.forward(xs[k]);
      const y    = ys[k];
      sum += -y * Math.log(pred + 1e-7) - (1 - y) * Math.log(1 - pred + 1e-7);
    }
    batchLosses.push(sum / BATCH);
  }

  const variance = arr => {
    const mean = arr.reduce((a, b) => a + b) / arr.length;
    return arr.reduce((s, v) => s + (v - mean) ** 2, 0) / arr.length;
  };
  const varSingle = variance(singleLosses);
  const varBatch  = variance(batchLosses);
  console.log(`  Single-sample variance:  ${varSingle.toFixed(5)}`);
  console.log(`  Batch-${BATCH} mean variance: ${varBatch.toFixed(5)}`);
  console.log(`  Ratio (expect ~1/${BATCH}):   ${(varBatch / varSingle).toFixed(3)}`);
  check(varBatch < varSingle / 4, `Batch mean variance is substantially lower`);
}

// ─────────────────────────────────────────────────────────────────
// Test 6: Balanced labels on a synthetic 50/50 dataset
//   The "all-red" bug would show up here as 50% accuracy
// ─────────────────────────────────────────────────────────────────
function testBalance() {
  console.log('\n[5] Sanity: balanced labels and non-trivial predictions');

  const net = new NN([2, 16, 1]);
  const N = 200;
  const xs = Array.from({ length: N }, (_, k) =>
    new Float64Array([(k % 10) / 5 - 1, Math.floor(k / 10) / 10 - 1])
  );
  // Alternate labels 1,0,1,0,...
  const ys = xs.map((_, i) => i % 2);

  for (let step = 0; step < 10000; step++) {
    const k = step % N;
    net.step(xs[k], ys[k], 0.05);
  }

  const preds = xs.map(x => net.forward(x));
  const min = Math.min(...preds).toFixed(4);
  const max = Math.max(...preds).toFixed(4);
  console.log(`  Prediction range: [${min}, ${max}]`);
  check(parseFloat(max) - parseFloat(min) > 0.2, 'Network outputs vary (not stuck at constant)');
}

// ─────────────────────────────────────────────────────────────────
// Run all tests
// ─────────────────────────────────────────────────────────────────
testGradients();
testLossDecreases();
testLinearSep();
testCircle();
testMiniBatchSmoother();
testBalance();

console.log('\nDone.\n');
