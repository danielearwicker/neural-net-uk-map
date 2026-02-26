// webgl-trainer.js — WebGL2 GPU-accelerated neural-network training.
//
// Implements the full training pipeline (forward, backprop, SGD weight update)
// as a sequence of fragment-shader draw calls.  All batch samples are processed
// in parallel on the GPU.
//
// Requires the EXT_color_buffer_float extension (rendering to R32F textures).
// Falls back gracefully if unavailable — the caller checks .ok before using.
//
// ── Texture layout ────────────────────────────────────────────────────────────
//
//   Training data  :  R32F  B × (numInputs+1)   row 0,1 = features, row 2 = label
//   Activations[l] :  R32F  B × sizes[l]        per-sample neuron activations
//   Deltas[l]      :  R32F  B × sizes[l+1]      per-sample error signals
//   Weights A/B    :  R32F  TEX_W × texH        ping-pong pair, same packing as renderer
//
// ── Per-step draw calls ───────────────────────────────────────────────────────
//
//   Forward pass        :  L   calls  (one per weight layer)
//   Output delta        :  1   call   (p − y for each sample)
//   Backward deltas     :  L−1 calls  (hidden error signals, reverse order)
//   Gradient + SGD      :  1   call   (each weight texel sums its own gradient
//                                       over the batch and applies the update)
//                  Total:  2L + 1
//
// ─────────────────────────────────────────────────────────────────────────────

class WebGLTrainer {
  /**
   * @param {WebGL2RenderingContext} gl    — shared context from the renderer
   * @param {WebGLVertexArrayObject}  vao  — full-screen-quad VAO from renderer
   */
  constructor(gl, vao) {
    this.gl  = gl;
    this.vao = vao;
    this.ok  = false;
    this.TEX_W = 256;

    // EXT_color_buffer_float is required to render into R32F textures.
    const ext = gl.getExtension('EXT_color_buffer_float');
    if (!ext) {
      console.warn('EXT_color_buffer_float not available — GPU training disabled');
      return;
    }
    this.ok = true;

    // Shared vertex shader (no v_uv — training passes use gl_FragCoord)
    this.vertShader = this._compile(gl.VERTEX_SHADER, `#version 300 es
in vec2 a_pos;
void main() { gl_Position = vec4(a_pos, 0.0, 1.0); }
`);

    // State initialised by buildForArchitecture()
    this.sizes    = null;
    this.L        = 0;
    this.texH     = 0;
    this.weightBuf = null;

    // Textures & FBOs
    this.trainDataTex = null;
    this.actTextures  = [];   // actTextures[l] for l = 1..L  (l=0 uses trainDataTex)
    this.actFBOs      = [];
    this.deltaTextures = [];  // deltaTextures[l] for l = 0..L-1
    this.deltaFBOs     = [];
    this.weightTexA = null;
    this.weightTexB = null;
    this.weightFBO_A = null;
    this.weightFBO_B = null;
    this.weightPhase = 0;     // 0 = A is current, 1 = B is current

    // Shader programs
    this.fwdPrograms  = [];   // one per layer
    this.fwdUniforms  = [];
    this.outDeltaProg = null;
    this.outDeltaUni  = null;
    this.bwdPrograms  = [];   // one per hidden layer (L-1 programs)
    this.bwdUniforms  = [];
    this.updateProg   = null;
    this.updateUni    = null;

    // Pre-allocated batch upload buffer (max batch = 128, 3 rows for 2 inputs + 1 label)
    this.maxBatch     = 128;
    this.batchBuf     = new Float32Array(this.maxBatch * 3);
  }

  // ═══════════════════════════════════════════════════════════════
  //  Public API
  // ═══════════════════════════════════════════════════════════════

  /**
   * (Re)build all shaders and textures for a network architecture.
   * Called on Reset / Resample.
   */
  buildForArchitecture(sizes, activation) {
    if (!this.ok) return;
    const gl = this.gl;
    this.sizes = sizes;
    this.L     = sizes.length - 1;
    const L    = this.L;

    // ── Dispose old resources ──
    this._disposeTraining();

    // ── Compute weight texture height ──
    let totalParams = 0;
    for (let l = 0; l < L; l++) {
      totalParams += sizes[l + 1] * sizes[l] + sizes[l + 1];
    }
    this.texH = Math.ceil(totalParams / this.TEX_W);
    this.weightBuf = new Float32Array(this.TEX_W * this.texH);

    // ── Allocate textures and FBOs ──

    // Training data: maxBatch × 3 (2 inputs + 1 label)
    this.trainDataTex = this._createTex(this.maxBatch, sizes[0] + 1);

    // Activation textures (l=1..L)
    this.actTextures = [null]; // index 0 unused (we use trainDataTex)
    this.actFBOs     = [null];
    for (let l = 1; l <= L; l++) {
      const tex = this._createTex(this.maxBatch, sizes[l]);
      this.actTextures.push(tex);
      this.actFBOs.push(this._createFBO(tex));
    }

    // Delta textures (l=0..L-1)
    this.deltaTextures = [];
    this.deltaFBOs     = [];
    for (let l = 0; l < L; l++) {
      const tex = this._createTex(this.maxBatch, sizes[l + 1]);
      this.deltaTextures.push(tex);
      this.deltaFBOs.push(this._createFBO(tex));
    }

    // Weight textures (ping-pong)
    this.weightTexA  = this._createTex(this.TEX_W, this.texH);
    this.weightTexB  = this._createTex(this.TEX_W, this.texH);
    this.weightFBO_A = this._createFBO(this.weightTexA);
    this.weightFBO_B = this._createFBO(this.weightTexB);
    this.weightPhase = 0;

    // ── Compute per-layer weight/bias offsets ──
    this.layerInfo = [];
    let off = 0;
    for (let l = 0; l < L; l++) {
      const nin = sizes[l], nout = sizes[l + 1];
      const wOff = off; off += nout * nin;
      const bOff = off; off += nout;
      this.layerInfo.push({ nin, nout, wOff, bOff });
    }

    // ── Build all shader programs ──
    this._buildForwardShaders(sizes, activation);
    this._buildOutputDeltaShader(sizes);
    this._buildBackwardShaders(sizes, activation);
    this._buildUpdateShader(sizes, activation);
  }

  /**
   * Upload initial weights from a CPU NN object to the current GPU weight texture.
   */
  uploadWeights(net) {
    if (!this.ok) return;
    const gl = this.gl;
    const buf = this.weightBuf;
    let off = 0;
    for (let l = 0; l < net.L; l++) {
      const Wl = net.W[l], bl = net.b[l];
      for (let j = 0; j < Wl.length; j++) {
        const row = Wl[j];
        for (let i = 0; i < row.length; i++) buf[off++] = row[i];
      }
      for (let j = 0; j < bl.length; j++) buf[off++] = bl[j];
    }
    while (off < buf.length) buf[off++] = 0;

    const tex = this.weightPhase === 0 ? this.weightTexA : this.weightTexB;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, this.TEX_W, this.texH, gl.RED, gl.FLOAT, buf);
  }

  /**
   * Download GPU weights back into a CPU NN object.
   * Used periodically for test-accuracy evaluation and when pausing training.
   */
  downloadWeights(net) {
    if (!this.ok) return;
    const gl = this.gl;
    const fbo = this.weightPhase === 0 ? this.weightFBO_A : this.weightFBO_B;
    const buf = this.weightBuf;

    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.readPixels(0, 0, this.TEX_W, this.texH, gl.RED, gl.FLOAT, buf);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    let off = 0;
    for (let l = 0; l < net.L; l++) {
      for (let j = 0; j < net.W[l].length; j++) {
        for (let i = 0; i < net.W[l][j].length; i++) {
          net.W[l][j][i] = buf[off++];
        }
      }
      for (let j = 0; j < net.b[l].length; j++) {
        net.b[l][j] = buf[off++];
      }
    }
  }

  /**
   * Upload a mini-batch of training samples to the GPU.
   * @param {Float64Array[]} xs  — array of B input vectors (each length 2)
   * @param {number[]}       ys  — array of B labels (0 or 1)
   * @param {number}         B   — batch size
   */
  uploadBatch(xs, ys, B) {
    if (!this.ok) return;
    const gl = this.gl;
    // Pack into trainDataTex: width=B, height=3 (row 0 = x0, row 1 = x1, row 2 = y)
    // texelFetch(tex, ivec2(sampleIdx, featureIdx), 0)
    // R32F texture stores one float per texel.
    // We need to upload row-by-row since the texture width is maxBatch but we only
    // fill the first B columns.
    const buf = this.batchBuf;
    // Row 0: feature 0
    for (let s = 0; s < B; s++) buf[s] = xs[s][0];
    // Row 1: feature 1
    for (let s = 0; s < B; s++) buf[this.maxBatch + s] = xs[s][1];
    // Row 2: labels
    for (let s = 0; s < B; s++) buf[2 * this.maxBatch + s] = ys[s];

    gl.bindTexture(gl.TEXTURE_2D, this.trainDataTex);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, this.maxBatch, 3, gl.RED, gl.FLOAT, buf);
  }

  /**
   * Execute one full training step on the GPU:
   * forward pass → output delta → backward deltas → gradient + SGD update.
   */
  trainStep(lr, B) {
    if (!this.ok) return;
    const gl = this.gl;
    const L  = this.L;

    const wRead = this.weightPhase === 0 ? this.weightTexA : this.weightTexB;
    const wFBO  = this.weightPhase === 0 ? this.weightFBO_B : this.weightFBO_A;

    // ── FORWARD PASS ──────────────────────────────────────────
    for (let l = 0; l < L; l++) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.actFBOs[l + 1]);
      gl.viewport(0, 0, B, this.sizes[l + 1]);
      gl.useProgram(this.fwdPrograms[l]);

      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, wRead);
      gl.uniform1i(this.fwdUniforms[l].u_weights, 0);

      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D,
        l === 0 ? this.trainDataTex : this.actTextures[l]);
      gl.uniform1i(this.fwdUniforms[l].u_aPrev, 1);

      gl.bindVertexArray(this.vao);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
    }

    // ── OUTPUT DELTA (p − y) ──────────────────────────────────
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.deltaFBOs[L - 1]);
    gl.viewport(0, 0, B, 1);
    gl.useProgram(this.outDeltaProg);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.actTextures[L]);
    gl.uniform1i(this.outDeltaUni.u_output, 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.trainDataTex);
    gl.uniform1i(this.outDeltaUni.u_trainData, 1);

    gl.bindVertexArray(this.vao);
    gl.drawArrays(gl.TRIANGLES, 0, 6);

    // ── BACKWARD DELTAS (hidden layers, reverse order) ────────
    for (let l = L - 2; l >= 0; l--) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.deltaFBOs[l]);
      gl.viewport(0, 0, B, this.sizes[l + 1]);
      gl.useProgram(this.bwdPrograms[l]);

      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, wRead);
      gl.uniform1i(this.bwdUniforms[l].u_weights, 0);

      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, this.deltaTextures[l + 1]);
      gl.uniform1i(this.bwdUniforms[l].u_deltaNext, 1);

      gl.activeTexture(gl.TEXTURE2);
      gl.bindTexture(gl.TEXTURE_2D, this.actTextures[l + 1]);
      gl.uniform1i(this.bwdUniforms[l].u_aCurrent, 2);

      gl.bindVertexArray(this.vao);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
    }

    // ── COMBINED GRADIENT ACCUMULATION + SGD UPDATE ───────────
    gl.bindFramebuffer(gl.FRAMEBUFFER, wFBO);
    gl.viewport(0, 0, this.TEX_W, this.texH);
    gl.useProgram(this.updateProg);

    let unit = 0;
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, wRead);
    gl.uniform1i(this.updateUni.u_weightsOld, unit++);

    for (let l = 0; l < L; l++) {
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_2D, this.deltaTextures[l]);
      gl.uniform1i(this.updateUni.u_deltas[l], unit++);
    }
    for (let l = 0; l < L; l++) {
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_2D,
        l === 0 ? this.trainDataTex : this.actTextures[l]);
      gl.uniform1i(this.updateUni.u_acts[l], unit++);
    }

    gl.uniform1f(this.updateUni.u_lr, lr);
    gl.uniform1i(this.updateUni.u_batchSize, B);

    gl.bindVertexArray(this.vao);
    gl.drawArrays(gl.TRIANGLES, 0, 6);

    // Flip ping-pong
    this.weightPhase = 1 - this.weightPhase;
  }

  /**
   * Read back the output activations and compute mean BCE loss on CPU.
   * @param {number} B — batch size
   * @returns {number} mean loss over the batch
   */
  readLoss(B) {
    if (!this.ok) return 0;
    const gl = this.gl;
    const L  = this.L;

    // Read output activations (B × 1)
    const preds = new Float32Array(this.maxBatch * 1);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.actFBOs[L]);
    gl.readPixels(0, 0, this.maxBatch, 1, gl.RED, gl.FLOAT, preds);

    // Read labels from training data texture (row 2)
    const labels = new Float32Array(this.maxBatch * 3);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    // We stored labels in batchBuf; just compute from batchBuf directly
    let totalLoss = 0;
    for (let s = 0; s < B; s++) {
      const p = preds[s];
      const y = this.batchBuf[2 * this.maxBatch + s];
      totalLoss += -y * Math.log(p + 1e-7) - (1 - y) * Math.log(1 - p + 1e-7);
    }
    return totalLoss / B;
  }

  /** Return the current weight texture (for the renderer to use directly). */
  currentWeightTexture() {
    return this.weightPhase === 0 ? this.weightTexA : this.weightTexB;
  }

  // ═══════════════════════════════════════════════════════════════
  //  Shader generation — Forward pass
  // ═══════════════════════════════════════════════════════════════

  _buildForwardShaders(sizes, activation) {
    const gl = this.gl;
    const L  = this.L;
    this.fwdPrograms = [];
    this.fwdUniforms = [];

    const actFn = activation === 'relu'
      ? 'float act(float z) { return max(0.0, z); }'
      : 'float act(float z) { return tanh(z); }';

    for (let l = 0; l < L; l++) {
      const { nin, nout, wOff, bOff } = this.layerInfo[l];
      const isLast = l === L - 1;
      const isFirst = l === 0;

      // For the output layer, use sigmoid; for hidden layers, use act()
      let src = `#version 300 es
precision highp float;
#define TEX_W ${this.TEX_W}
#define NIN ${nin}
uniform sampler2D u_weights;
uniform sampler2D u_aPrev;
float w(int idx) { return texelFetch(u_weights, ivec2(idx % TEX_W, idx / TEX_W), 0).r; }
${actFn}
float sig(float z) { return 1.0 / (1.0 + exp(-clamp(z, -500.0, 500.0))); }
out vec4 fragColor;
void main() {
  int s = int(gl_FragCoord.x);
  int j = int(gl_FragCoord.y);
  float z = w(${bOff} + j);
  int wBase = ${wOff} + j * NIN;
  for (int i = 0; i < NIN; i++) {
    float ai = texelFetch(u_aPrev, ivec2(s, i), 0).r;
    z += w(wBase + i) * ai;
  }
  fragColor = vec4(${isLast ? 'sig(z)' : 'act(z)'}, 0.0, 0.0, 1.0);
}
`;
      const prog = this._link(src);
      this.fwdPrograms.push(prog);
      this.fwdUniforms.push({
        u_weights: gl.getUniformLocation(prog, 'u_weights'),
        u_aPrev:   gl.getUniformLocation(prog, 'u_aPrev'),
      });
    }
  }

  // ═══════════════════════════════════════════════════════════════
  //  Shader generation — Output delta
  // ═══════════════════════════════════════════════════════════════

  _buildOutputDeltaShader(sizes) {
    const gl = this.gl;
    const labelRow = sizes[0]; // = 2 (labels are in row 2 of trainDataTex)

    const src = `#version 300 es
precision highp float;
uniform sampler2D u_output;
uniform sampler2D u_trainData;
out vec4 fragColor;
void main() {
  int s = int(gl_FragCoord.x);
  float p = texelFetch(u_output, ivec2(s, 0), 0).r;
  float y = texelFetch(u_trainData, ivec2(s, ${labelRow}), 0).r;
  fragColor = vec4(p - y, 0.0, 0.0, 1.0);
}
`;
    const prog = this._link(src);
    this.outDeltaProg = prog;
    this.outDeltaUni  = {
      u_output:    gl.getUniformLocation(prog, 'u_output'),
      u_trainData: gl.getUniformLocation(prog, 'u_trainData'),
    };
  }

  // ═══════════════════════════════════════════════════════════════
  //  Shader generation — Backward deltas (hidden layers)
  // ═══════════════════════════════════════════════════════════════

  _buildBackwardShaders(sizes, activation) {
    const gl = this.gl;
    const L  = this.L;
    this.bwdPrograms = [];
    this.bwdUniforms = [];

    const derivFn = activation === 'relu'
      ? 'float actDeriv(float a) { return a > 0.0 ? 1.0 : 0.0; }'
      : 'float actDeriv(float a) { return 1.0 - a * a; }';

    // We need backward delta shaders for layers l = 0 .. L-2.
    // delta[l] is computed from delta[l+1] and W[l+1].
    for (let l = 0; l < L - 1; l++) {
      const ninNext  = sizes[l + 1]; // = this delta's width
      const noutNext = sizes[l + 2]; // = delta[l+1]'s width
      const { wOff: wOffNext } = this.layerInfo[l + 1];

      const src = `#version 300 es
precision highp float;
#define TEX_W ${this.TEX_W}
#define NIN_NEXT ${ninNext}
#define NOUT_NEXT ${noutNext}
uniform sampler2D u_weights;
uniform sampler2D u_deltaNext;
uniform sampler2D u_aCurrent;
float w(int idx) { return texelFetch(u_weights, ivec2(idx % TEX_W, idx / TEX_W), 0).r; }
${derivFn}
out vec4 fragColor;
void main() {
  int s = int(gl_FragCoord.x);
  int i = int(gl_FragCoord.y);
  float sumWD = 0.0;
  for (int j = 0; j < NOUT_NEXT; j++) {
    float wji = w(${wOffNext} + j * NIN_NEXT + i);
    float dj  = texelFetch(u_deltaNext, ivec2(s, j), 0).r;
    sumWD += wji * dj;
  }
  float a = texelFetch(u_aCurrent, ivec2(s, i), 0).r;
  fragColor = vec4(sumWD * actDeriv(a), 0.0, 0.0, 1.0);
}
`;
      const prog = this._link(src);
      this.bwdPrograms.push(prog);
      this.bwdUniforms.push({
        u_weights:   gl.getUniformLocation(prog, 'u_weights'),
        u_deltaNext: gl.getUniformLocation(prog, 'u_deltaNext'),
        u_aCurrent:  gl.getUniformLocation(prog, 'u_aCurrent'),
      });
    }
  }

  // ═══════════════════════════════════════════════════════════════
  //  Shader generation — Combined gradient + SGD weight update
  // ═══════════════════════════════════════════════════════════════

  _buildUpdateShader(sizes, activation) {
    const gl = this.gl;
    const L  = this.L;

    // Build uniform declarations for delta and activation textures
    let uniformDecls = 'uniform sampler2D u_weightsOld;\n';
    for (let l = 0; l < L; l++) uniformDecls += `uniform sampler2D u_delta${l};\n`;
    for (let l = 0; l < L; l++) uniformDecls += `uniform sampler2D u_act${l};\n`;
    uniformDecls += 'uniform float u_lr;\n';
    uniformDecls += 'uniform int u_batchSize;\n';

    // Build the per-layer gradient computation blocks.
    // Each fragment computes the mean gradient for ONE weight/bias and applies SGD.
    let layerBlocks = '';
    for (let l = 0; l < L; l++) {
      const { nin, nout, wOff, bOff } = this.layerInfo[l];
      const layerEnd = bOff + nout; // first index PAST this layer's params

      const condition = l === 0
        ? `if (flatIdx < ${layerEnd})`
        : `else if (flatIdx < ${layerEnd})`;

      layerBlocks += `  ${condition} {\n`;
      layerBlocks += `    int localIdx = flatIdx - ${wOff};\n`;
      layerBlocks += `    if (localIdx < ${nout * nin}) {\n`;
      layerBlocks += `      int j = localIdx / ${nin};\n`;
      layerBlocks += `      int i = localIdx - j * ${nin};\n`;
      layerBlocks += `      for (int s = 0; s < u_batchSize; s++) {\n`;
      layerBlocks += `        float dj = texelFetch(u_delta${l}, ivec2(s, j), 0).r;\n`;
      layerBlocks += `        float ai = texelFetch(u_act${l}, ivec2(s, i), 0).r;\n`;
      layerBlocks += `        gradSum += dj * ai;\n`;
      layerBlocks += `      }\n`;
      layerBlocks += `    } else {\n`;
      layerBlocks += `      int j = localIdx - ${nout * nin};\n`;
      layerBlocks += `      for (int s = 0; s < u_batchSize; s++) {\n`;
      layerBlocks += `        gradSum += texelFetch(u_delta${l}, ivec2(s, j), 0).r;\n`;
      layerBlocks += `      }\n`;
      layerBlocks += `    }\n`;
      layerBlocks += `  }\n`;
    }

    const src = `#version 300 es
precision highp float;
#define TEX_W ${this.TEX_W}
${uniformDecls}
out vec4 fragColor;
void main() {
  int flatIdx = int(gl_FragCoord.y) * TEX_W + int(gl_FragCoord.x);
  float wOld = texelFetch(u_weightsOld, ivec2(gl_FragCoord.xy), 0).r;
  float gradSum = 0.0;
${layerBlocks}
  float meanGrad = gradSum / float(u_batchSize);
  fragColor = vec4(wOld - u_lr * meanGrad, 0.0, 0.0, 1.0);
}
`;

    const prog = this._link(src);
    this.updateProg = prog;

    // Collect uniform locations
    const uni = {
      u_weightsOld: gl.getUniformLocation(prog, 'u_weightsOld'),
      u_lr:         gl.getUniformLocation(prog, 'u_lr'),
      u_batchSize:  gl.getUniformLocation(prog, 'u_batchSize'),
      u_deltas: [],
      u_acts:   [],
    };
    for (let l = 0; l < L; l++) {
      uni.u_deltas.push(gl.getUniformLocation(prog, `u_delta${l}`));
      uni.u_acts.push(gl.getUniformLocation(prog, `u_act${l}`));
    }
    this.updateUni = uni;
  }

  // ═══════════════════════════════════════════════════════════════
  //  Low-level helpers
  // ═══════════════════════════════════════════════════════════════

  _createTex(width, height) {
    const gl = this.gl;
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, width, height, 0, gl.RED, gl.FLOAT, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return tex;
  }

  _createFBO(texture) {
    const gl = this.gl;
    const fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (status !== gl.FRAMEBUFFER_COMPLETE) {
      console.error('Framebuffer incomplete:', status);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return fbo;
  }

  _compile(type, src) {
    const gl = this.gl;
    const s  = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      const log = gl.getShaderInfoLog(s);
      gl.deleteShader(s);
      throw new Error('Shader compile error:\n' + log + '\n--- source ---\n' + src);
    }
    return s;
  }

  _link(fragSrc) {
    const gl = this.gl;
    const frag = this._compile(gl.FRAGMENT_SHADER, fragSrc);
    const prog = gl.createProgram();
    gl.attachShader(prog, this.vertShader);
    gl.attachShader(prog, frag);
    gl.bindAttribLocation(prog, 0, 'a_pos');
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      console.error('Link error:', gl.getProgramInfoLog(prog));
    }
    gl.deleteShader(frag);
    return prog;
  }

  _disposeTraining() {
    const gl = this.gl;
    const del = (arr, fn) => { for (const x of arr) if (x) fn.call(gl, x); };
    del(this.actTextures,  gl.deleteTexture);
    del(this.actFBOs,      gl.deleteFramebuffer);
    del(this.deltaTextures, gl.deleteTexture);
    del(this.deltaFBOs,    gl.deleteFramebuffer);
    del(this.fwdPrograms,  gl.deleteProgram);
    del(this.bwdPrograms,  gl.deleteProgram);
    if (this.outDeltaProg) gl.deleteProgram(this.outDeltaProg);
    if (this.updateProg)   gl.deleteProgram(this.updateProg);
    if (this.weightTexA)   gl.deleteTexture(this.weightTexA);
    if (this.weightTexB)   gl.deleteTexture(this.weightTexB);
    if (this.weightFBO_A)  gl.deleteFramebuffer(this.weightFBO_A);
    if (this.weightFBO_B)  gl.deleteFramebuffer(this.weightFBO_B);
    if (this.trainDataTex) gl.deleteTexture(this.trainDataTex);
  }
}

if (typeof module !== 'undefined') module.exports = { WebGLTrainer };
