// webgl-renderer.js — WebGL2-accelerated neural-network visualiser.
//
// Evaluates the network per-pixel on the GPU via a generated fragment shader.
// The fragment shader is regenerated whenever the network architecture changes
// (depth, width, activation), baking exact layer sizes as compile-time constants
// for tight inner loops.  Weights are uploaded each frame as an R32F texture.
//
// Training stays in JS (nn.js is unchanged).

class WebGLNetRenderer {
  /**
   * @param {HTMLCanvasElement} canvas  — the prediction canvas
   * @param {number} width              — canvas pixel width
   * @param {number} height             — canvas pixel height
   */
  constructor(canvas, width, height) {
    this.width  = width;
    this.height = height;
    this.TEX_W  = 256;           // weight-texture width (texels per row)

    const gl = canvas.getContext('webgl2', {
      antialias: false, depth: false, stencil: false,
      premultipliedAlpha: false,
    });
    if (!gl) throw new Error('WebGL2 not available');
    this.gl = gl;

    // ── Static vertex shader (full-screen quad) ──────────────────
    this.vertShader = this._compile(gl.VERTEX_SHADER, `#version 300 es
in vec2 a_pos;
out vec2 v_uv;
void main() {
  v_uv = a_pos * 0.5 + 0.5;
  gl_Position = vec4(a_pos, 0.0, 1.0);
}`);

    // ── Full-screen quad (two triangles) ─────────────────────────
    const vbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
      -1, -1,  1, -1,  -1, 1,
      -1,  1,  1, -1,   1, 1,
    ]), gl.STATIC_DRAW);

    this.vao = gl.createVertexArray();
    gl.bindVertexArray(this.vao);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);

    // ── Weight texture (R32F — one float per texel) ──────────────
    this.weightTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.weightTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    // ── Map overlay texture (RGBA8) ──────────────────────────────
    this.mapTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.mapTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    this.program   = null;
    this.uWeights  = null;
    this.uMap      = null;
    this.weightBuf = null;   // pre-allocated Float32Array for packing
    this.texH      = 0;     // current weight-texture height

    gl.viewport(0, 0, width, height);
  }

  // ─────────────────────────────────────────────────────────────
  // Recompile the shader for a (possibly new) architecture.
  // Called on Reset / Resample — not per frame.
  // ─────────────────────────────────────────────────────────────
  buildShader(sizes, activation) {
    const gl = this.gl;

    // Generate and compile the fragment shader
    const fragSrc = this._fragSource(sizes, activation);
    const frag    = this._compile(gl.FRAGMENT_SHADER, fragSrc);

    // Link program
    if (this.program) gl.deleteProgram(this.program);
    this.program = gl.createProgram();
    gl.attachShader(this.program, this.vertShader);
    gl.attachShader(this.program, frag);
    gl.bindAttribLocation(this.program, 0, 'a_pos');
    gl.linkProgram(this.program);
    if (!gl.getProgramParameter(this.program, gl.LINK_STATUS)) {
      console.error('Shader link error:', gl.getProgramInfoLog(this.program));
    }
    gl.deleteShader(frag);

    this.uWeights = gl.getUniformLocation(this.program, 'u_weights');
    this.uMap     = gl.getUniformLocation(this.program, 'u_map');

    // Pre-allocate the weight-packing buffer (reused every frame)
    let total = 0;
    for (let l = 0; l < sizes.length - 1; l++) {
      total += sizes[l + 1] * sizes[l] + sizes[l + 1];
    }
    this.texH      = Math.ceil(total / this.TEX_W);
    this.weightBuf = new Float32Array(this.TEX_W * this.texH);

    // Allocate (or resize) the weight texture
    gl.bindTexture(gl.TEXTURE_2D, this.weightTex);
    gl.texImage2D(
      gl.TEXTURE_2D, 0, gl.R32F,
      this.TEX_W, this.texH, 0,
      gl.RED, gl.FLOAT, null,
    );
  }

  // ─────────────────────────────────────────────────────────────
  // Upload the map canvas (with sample dots) as the overlay texture.
  // Called once on load and again on each Resample.
  // ─────────────────────────────────────────────────────────────
  uploadMap(mapCanvas) {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.mapTex);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, mapCanvas);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
  }

  // ─────────────────────────────────────────────────────────────
  // Pack current weights, upload to GPU, and draw one frame.
  // Called every animation frame during training.
  // ─────────────────────────────────────────────────────────────
  render(net) {
    if (!this.program || !net) return;
    const gl = this.gl;

    // ── Pack weights into the pre-allocated Float32Array ──
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
    // Zero-pad the remainder (needed only after architecture change,
    // but cheap enough to do every frame for simplicity)
    while (off < buf.length) buf[off++] = 0;

    // ── Upload weight texture ────────────────────────────────
    gl.bindTexture(gl.TEXTURE_2D, this.weightTex);
    gl.texSubImage2D(
      gl.TEXTURE_2D, 0, 0, 0,
      this.TEX_W, this.texH,
      gl.RED, gl.FLOAT, buf,
    );

    // ── Draw ─────────────────────────────────────────────────
    gl.useProgram(this.program);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.weightTex);
    gl.uniform1i(this.uWeights, 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.mapTex);
    gl.uniform1i(this.uMap, 1);

    gl.bindVertexArray(this.vao);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
  }

  // ═══════════════════════════════════════════════════════════
  //  Private helpers
  // ═══════════════════════════════════════════════════════════

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

  // ─────────────────────────────────────────────────────────────
  // Generate the fragment shader source for a given architecture.
  //
  // The shader evaluates the full forward pass per-pixel:
  //   input (x,y) → hidden layers (tanh/ReLU) → sigmoid output → colour.
  //
  // Weights are read from an R32F texture via integer-indexed texelFetch.
  // Two float arrays (buf0, buf1) ping-pong as source/destination
  // across layers so we never need more than 2 × maxWidth floats.
  // ─────────────────────────────────────────────────────────────
  _fragSource(sizes, activation) {
    const L      = sizes.length - 1;              // number of weight layers
    const hidden = sizes.slice(1, -1);            // hidden layer widths
    const maxW   = hidden.length ? Math.max(...hidden) : 1;

    const actLine = activation === 'relu'
      ? 'float act(float z) { return max(0.0, z); }'
      : 'float act(float z) { return tanh(z); }';

    // ── Compute per-layer weight / bias offsets in the texture ──
    let offset = 0;
    const layers = [];
    for (let l = 0; l < L; l++) {
      const nin = sizes[l], nout = sizes[l + 1];
      const wOff = offset;  offset += nout * nin;
      const bOff = offset;  offset += nout;
      layers.push({ nin, nout, wOff, bOff });
    }

    // ── Generate per-layer GLSL ─────────────────────────────────
    //
    // Ping-pong convention:
    //   Layer l writes to buf[l % 2].
    //   Layer l reads from buf[(l+1) % 2]  (for l > 0).
    //   Layer 0 reads from (in0, in1).
    //
    let body = '';
    for (let l = 0; l < L; l++) {
      const { nin, nout, wOff, bOff } = layers[l];
      const isFirst = l === 0;
      const isLast  = l === L - 1;
      const dst = 'buf' + (l % 2);
      const src = 'buf' + ((l + 1) % 2);

      if (isLast) {
        // ── Output layer → float p ──
        body += '  // Output layer ' + l + ': ' + nin + ' -> 1\n';
        body += '  {\n';
        body += '    int wo = ' + wOff + ';\n';
        body += '    float z = w(' + bOff + ');\n';
        if (isFirst) {
          // Degenerate case: input feeds directly into the output
          body += '    z += w(wo) * in0; wo++;\n';
          body += '    z += w(wo) * in1; wo++;\n';
        } else {
          body += '    for (int i = 0; i < ' + nin + '; i++) { z += w(wo) * ' + src + '[i]; wo++; }\n';
        }
        body += '    p = sig(z);\n';
        body += '  }\n';

      } else if (isFirst) {
        // ── First hidden layer (reads 2D input) ──
        body += '  // Layer 0: input(' + nin + ') -> ' + dst + '[' + nout + ']\n';
        body += '  {\n';
        body += '    int wo = ' + wOff + ';\n';
        body += '    for (int j = 0; j < ' + nout + '; j++) {\n';
        body += '      float z = w(' + bOff + ' + j);\n';
        body += '      z += w(wo) * in0; wo++;\n';
        body += '      z += w(wo) * in1; wo++;\n';
        body += '      ' + dst + '[j] = act(z);\n';
        body += '    }\n';
        body += '  }\n';

      } else {
        // ── Middle hidden layer ──
        body += '  // Layer ' + l + ': ' + src + '[' + nin + '] -> ' + dst + '[' + nout + ']\n';
        body += '  {\n';
        body += '    int wo = ' + wOff + ';\n';
        body += '    for (int j = 0; j < ' + nout + '; j++) {\n';
        body += '      float z = w(' + bOff + ' + j);\n';
        body += '      for (int i = 0; i < ' + nin + '; i++) { z += w(wo) * ' + src + '[i]; wo++; }\n';
        body += '      ' + dst + '[j] = act(z);\n';
        body += '    }\n';
        body += '  }\n';
      }
    }

    // ── Assemble the full shader ────────────────────────────────
    return '#version 300 es\n'
      + 'precision highp float;\n'
      + '\n'
      + '#define TW ' + this.TEX_W + '\n'
      + '\n'
      + 'uniform sampler2D u_weights;\n'
      + 'uniform sampler2D u_map;\n'
      + 'in  vec2 v_uv;\n'
      + 'out vec4 fragColor;\n'
      + '\n'
      + '// Read one float from the weight texture by linear index.\n'
      + 'float w(int idx) {\n'
      + '  return texelFetch(u_weights, ivec2(idx % TW, idx / TW), 0).r;\n'
      + '}\n'
      + '\n'
      + actLine + '\n'
      + '\n'
      + 'float sig(float z) {\n'
      + '  return 1.0 / (1.0 + exp(-clamp(z, -500.0, 500.0)));\n'
      + '}\n'
      + '\n'
      + 'void main() {\n'
      + '  // Map UV to network input range [-1, +1].\n'
      + '  // v_uv.y is bottom-to-top in GL; the network expects top-to-bottom.\n'
      + '  float in0 = v_uv.x * 2.0 - 1.0;\n'
      + '  float in1 = (1.0 - v_uv.y) * 2.0 - 1.0;\n'
      + '\n'
      + '  float buf0[' + maxW + '];\n'
      + '  float buf1[' + maxW + '];\n'
      + '  float p;\n'
      + '\n'
      + body
      + '\n'
      + '  // Colour scheme (matches the JS version exactly):\n'
      + '  //   outside UK = red (#ef4444), inside UK = blue (#3b82f6),\n'
      + '  //   uncertain boundary region fades toward white.\n'
      + '  float unc = 1.0 - abs(p - 0.5) * 2.0;\n'
      + '  vec3 netColor = clamp(vec3(\n'
      + '    (239.0 * (1.0 - p) + 255.0 * unc) / 255.0,\n'
      + '    ( 68.0 * (1.0 - p) + 255.0 * unc) / 255.0,\n'
      + '    (246.0 * p         + 255.0 * unc) / 255.0\n'
      + '  ), 0.0, 1.0);\n'
      + '\n'
      + '  // Overlay the map (with sample dots) at 20% opacity.\n'
      + '  vec3 mapColor = texture(u_map, v_uv).rgb;\n'
      + '  fragColor = vec4(mix(netColor, mapColor, 0.2), 1.0);\n'
      + '}\n';
  }
}

// Export for Node.js (allows basic smoke-testing of shader generation)
if (typeof module !== 'undefined') module.exports = { WebGLNetRenderer };
