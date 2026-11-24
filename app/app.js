/* Radar YOLO – Tarea 7 Redes Neuronales (UNISON)
 * Detecta vehículos y placas con un YOLOv8 entrenado en La Yuca (exportado a ONNX),
 * mide la velocidad entre dos líneas virtuales y, si hay exceso, genera una multa
 * SIMULADA con la placa leída por OCR (Tesseract.js).
 */
'use strict';

/* =========================================================================
 * Núcleo (funciones puras, se pueden probar en Node)
 * ========================================================================= */
const Core = (() => {
  function iou(a, b) {
    const x1 = Math.max(a[0], b[0]), y1 = Math.max(a[1], b[1]);
    const x2 = Math.min(a[2], b[2]), y2 = Math.min(a[3], b[3]);
    const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    const ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
    return ua > 0 ? inter / ua : 0;
  }

  // NMS por clase
  function nms(dets, thr = 0.45) {
    const out = [];
    const porClase = {};
    for (const d of dets) (porClase[d.cls] ||= []).push(d);
    for (const k in porClase) {
      const arr = porClase[k].sort((a, b) => b.score - a.score);
      const keep = [];
      for (const d of arr) {
        if (keep.every(q => iou(q.box, d.box) < thr)) keep.push(d);
      }
      out.push(...keep);
    }
    return out;
  }

  /* Decodifica la salida de YOLOv8 exportado a ONNX.
   * Forma típica: [1, 4+nc, N] (cx, cy, w, h, score_c...). También acepta [1, N, 4+nc].
   * lb = {scale, padX, padY, W, H} para deshacer el letterbox. */
  function decode(data, dims, nc, conf, lb) {
    let C = dims[1], N = dims[2], transpuesto = false;
    if (C !== 4 + nc && N === 4 + nc) { transpuesto = true; [C, N] = [N, C]; }
    const at = transpuesto ? (c, i) => data[i * C + c] : (c, i) => data[c * N + i];
    const dets = [];
    for (let i = 0; i < N; i++) {
      let best = 0, cls = -1;
      for (let c = 0; c < nc; c++) {
        const s = at(4 + c, i);
        if (s > best) { best = s; cls = c; }
      }
      if (best < conf) continue;
      const cx = at(0, i), cy = at(1, i), w = at(2, i), h = at(3, i);
      let x1 = (cx - w / 2 - lb.padX) / lb.scale, y1 = (cy - h / 2 - lb.padY) / lb.scale;
      let x2 = (cx + w / 2 - lb.padX) / lb.scale, y2 = (cy + h / 2 - lb.padY) / lb.scale;
      x1 = Math.max(0, x1); y1 = Math.max(0, y1);
      x2 = Math.min(lb.W, x2); y2 = Math.min(lb.H, y2);
      if (x2 - x1 < 2 || y2 - y1 < 2) continue;
      dets.push({ box: [x1, y1, x2, y2], score: best, cls });
    }
    return nms(dets);
  }

  function letterbox(W, H, S) {
    const scale = Math.min(S / W, S / H);
    const nw = Math.round(W * scale), nh = Math.round(H * scale);
    return { scale, padX: Math.floor((S - nw) / 2), padY: Math.floor((S - nh) / 2), nw, nh, W, H };
  }

  // Tiempo interpolado en que la posición cruzó la línea L (o null si no cruzó)
  function cruce(pPrev, tPrev, p, t, L) {
    if (pPrev == null || pPrev === p) return null;
    if ((pPrev - L) * (p - L) > 0) return null;
    return tPrev + (L - pPrev) / (p - pPrev) * (t - tPrev);
  }

  const centro = b => [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2];

  /* Tracker sencillo por IoU + distancia de centroides, con medición de velocidad */
  class Tracker {
    constructor() { this.tracks = []; this.nextId = 1; this.medidos = 0; }
    reset() { this.tracks = []; }

    /* dets: detecciones del vehículo; t: segundos; cfg: {orient, LA, LB (px), dist (m), maxAge}
     * Devuelve los tracks que acaban de obtener velocidad en esta llamada. */
    update(dets, t, cfg) {
      const nuevos = [];
      const pares = [];
      this.tracks.forEach((tr, i) => dets.forEach((d, j) => {
        const v = iou(tr.box, d.box);
        if (v > 0.1) pares.push([v, i, j]);
      }));
      pares.sort((a, b) => b[0] - a[0]);
      const usadosT = new Set(), usadosD = new Set(), asign = [];
      for (const [, i, j] of pares) {
        if (usadosT.has(i) || usadosD.has(j)) continue;
        usadosT.add(i); usadosD.add(j); asign.push([i, j]);
      }
      // respaldo: centroide más cercano (cuando los FPS son bajos y no hay traslape)
      this.tracks.forEach((tr, i) => {
        if (usadosT.has(i)) return;
        const [tx, ty] = centro(tr.box);
        const lim = Math.max(tr.box[2] - tr.box[0], tr.box[3] - tr.box[1]) * 1.2;
        let mejor = -1, md = lim;
        dets.forEach((d, j) => {
          if (usadosD.has(j) || d.cls !== tr.cls) return;
          const [dx, dy] = centro(d.box);
          const dd = Math.hypot(dx - tx, dy - ty);
          if (dd < md) { md = dd; mejor = j; }
        });
        if (mejor >= 0) { usadosT.add(i); usadosD.add(mejor); asign.push([i, mejor]); }
      });

      for (const [i, j] of asign) {
        const tr = this.tracks[i], d = dets[j];
        tr.box = d.box; tr.score = d.score; tr.cls = d.cls; tr.lastSeen = t; tr.hits++;
        const p = this._pos(d.box, cfg.orient);
        const cA = cruce(tr.pos, tr.t, p, t, cfg.LA);
        const cB = cruce(tr.pos, tr.t, p, t, cfg.LB);
        if (cA != null && tr.tA == null) tr.tA = cA;
        if (cB != null && tr.tB == null) tr.tB = cB;
        tr.pos = p; tr.t = t;
        if (tr.speed == null && tr.tA != null && tr.tB != null) {
          const dt = Math.abs(tr.tB - tr.tA);
          if (dt > 0.03) {
            tr.speed = cfg.dist / dt * 3.6; // km/h
            tr.dt = dt;
            this.medidos++;
            nuevos.push(tr);
          }
        }
      }
      dets.forEach((d, j) => {
        if (usadosD.has(j)) return;
        this.tracks.push({
          id: this.nextId++, box: d.box, score: d.score, cls: d.cls,
          pos: this._pos(d.box, cfg.orient), t, lastSeen: t, hits: 1,
          tA: null, tB: null, speed: null, plate: null, multado: false,
        });
      });
      this.tracks = this.tracks.filter(tr => t - tr.lastSeen <= cfg.maxAge && t >= tr.lastSeen - 0.5);
      return nuevos;
    }
    _pos(b, orient) { const [cx, cy] = centro(b); return orient === 'h' ? cx : cy; }
  }

  function limpiarPlaca(txt) {
    return (txt || '').toUpperCase().replace(/[^A-Z0-9-]/g, '').replace(/^-+|-+$/g, '');
  }

  // Formato Sonora/México: LL-DDD-L (también 3 letras / 3-4 dígitos)
  const D2L = { '0': 'O', '1': 'I', '2': 'Z', '3': 'B', '4': 'A', '5': 'S', '6': 'G', '7': 'T', '8': 'B' };
  const L2D = { O: '0', Q: '0', D: '0', I: '1', L: '1', Z: '2', S: '5', G: '6', B: '8', T: '7' };
  const RE_PLACA = /^[A-Z]{2,3}-?\d{2,4}-?[A-Z]{1,2}$/;

  // Intenta ajustar un texto OCR al formato LL-DDD-L corrigiendo confusiones letra/dígito
  function normalizarPlaca(txt) {
    const t = limpiarPlaca(txt).replace(/-/g, '');
    if (t.length < 5) return limpiarPlaca(txt);
    // patrón 2 letras + 3 dígitos + 1 letra (el más común) y variantes de longitud
    const pats = [['LL', 'DDD', 'L'], ['LLL', 'DDD', 'L'], ['LL', 'DDD', 'LL'], ['LLL', 'DD', 'L'], ['LL', 'DDDD', 'L']];
    let mejor = null;
    for (const p of pats) {
      const pat = p.join('');
      for (let off = 0; off + pat.length <= t.length; off++) {
        if (off > 1 || t.length - off - pat.length > 1) continue;
        let out = '', cambios = 0;
        for (let i = 0; i < pat.length; i++) {
          const ch = t[off + i];
          const esL = /[A-Z]/.test(ch), quiereL = pat[i] === 'L';
          if (esL === quiereL) out += ch;
          else {
            const r = quiereL ? D2L[ch] : L2D[ch];
            if (!r) { out = null; break; }
            out += r; cambios++;
          }
        }
        if (!out) continue;
        const c = cambios + off + (t.length - off - pat.length);
        if (!mejor || c < mejor.c) {
          let a = p[0].length, b = a + p[1].length;
          mejor = { c, v: out.slice(0, a) + '-' + out.slice(a, b) + '-' + out.slice(b) };
        }
      }
    }
    return mejor && mejor.c <= 2 ? mejor.v : limpiarPlaca(txt);
  }

  // Puntaje de un candidato OCR: cumple formato > confianza > longitud razonable
  function puntuarPlaca(txt, conf) {
    const n = normalizarPlaca(txt);
    let p = (conf || 0) / 100;
    if (RE_PLACA.test(n)) p += 2;
    const len = n.replace(/-/g, '').length;
    if (len >= 5 && len <= 7) p += 0.5; else p -= 0.5 * Math.abs(len - 6);
    return { txt: n, score: p };
  }

  function multaMonto(vel, lim) {
    const exceso = Math.max(0, vel - lim);
    return Math.round(800 + exceso * 45);
  }

  return { iou, nms, decode, letterbox, cruce, Tracker, centro, limpiarPlaca, normalizarPlaca, puntuarPlaca, multaMonto };
})();

if (typeof module !== 'undefined') module.exports = Core;

/* =========================================================================
 * Aplicación (navegador)
 * ========================================================================= */
if (typeof window !== 'undefined') (() => {
  const $ = id => document.getElementById(id);
  const video = $('video'), view = $('view'), vctx = view.getContext('2d');
  const frame = document.createElement('canvas'), fctx = frame.getContext('2d', { willReadFrequently: true });
  const inp = document.createElement('canvas'), ictx = inp.getContext('2d', { willReadFrequently: true });

  let session = null, meta = null, S = 640, nc = 1, nombres = {};
  let tensorBuf = null;
  let corriendo = false, ocupado = false, stream = null;
  let ultDets = [], fpsEMA = 0, multas = 0, folio = 0;
  const tracker = new Core.Tracker();
  let ocrWorker = null, ocrListo = null;

  const log = m => { const l = $('log'); l.textContent += m + '\n'; l.scrollTop = 1e9; console.log(m); };
  const estado = (m, color) => { const e = $('estado'); e.textContent = m; e.style.color = color || ''; };

  /* ---------- Carga del modelo ---------- */
  async function cargarModelo() {
    try {
      ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';
      ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 1);
      meta = await (await fetch('model/meta.json', { cache: 'no-cache' })).json();
      S = meta.imgsz || 640;
      nombres = meta.names;
      nc = Object.keys(nombres).length;
      log('Clases: ' + JSON.stringify(nombres) + ' | imgsz ' + S);
      session = await ort.InferenceSession.create('model/best.onnx', {
        executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
      log('Entradas: ' + session.inputNames + ' | Salidas: ' + session.outputNames);
      inp.width = inp.height = S;
      tensorBuf = new Float32Array(3 * S * S);
      llenarClases();
      $('btnCam').disabled = false;
      estado('modelo listo', 'var(--ok)');
      prepararOCR();
    } catch (e) {
      estado('error al cargar modelo', 'var(--bad)');
      log('ERROR: ' + e.message + '\n¿Subiste model/best.onnx y model/meta.json?');
    }
  }

  function llenarClases() {
    const sv = $('claseVeh'), sp = $('clasePlaca');
    const ids = Object.keys(nombres).map(Number);
    const esPlaca = n => /plate|placa|licen|matr|number/i.test(n);
    const esVeh = n => /veh|car|auto|truck|bus|moto|van|coche|camion/i.test(n) && !esPlaca(n);
    sp.innerHTML = ids.map(i => `<option value="${i}">${nombres[i]}</option>`).join('');
    sv.innerHTML = `<option value="*">todas excepto la placa</option>` +
      ids.map(i => `<option value="${i}">${nombres[i]}</option>`).join('');
    const placa = ids.find(i => esPlaca(nombres[i]));
    if (placa != null) sp.value = placa;
    const veh = ids.filter(i => esVeh(nombres[i]));
    if (veh.length === 1) sv.value = veh[0];
    else if (veh.length > 1 || ids.length > 1) sv.value = '*';
    else sv.value = ids[0]; // modelo de una sola clase: se rastrea la placa misma
  }

  function prepararOCR() {
    if (ocrListo) return ocrListo;
    ocrListo = (async () => {
      ocrWorker = await Tesseract.createWorker('eng');
      await ocrWorker.setParameters({
        tessedit_char_whitelist: 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-',
        tessedit_pageseg_mode: '7',
      });
      log('OCR listo');
    })().catch(e => { log('OCR no disponible: ' + e.message); ocrWorker = null; });
    return ocrListo;
  }

  /* ---------- Fuentes de video ---------- */
  async function iniciarCamara() {
    detener();
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 720 } },
      });
      video.srcObject = stream;
      await video.play();
      arrancar('cámara');
    } catch (e) { log('No se pudo abrir la cámara: ' + e.message); }
  }

  async function iniciarVideo(file) {
    detener();
    video.srcObject = null;
    video.src = URL.createObjectURL(file);
    video.loop = false;
    video.playbackRate = parseFloat($('rate').value);
    await video.play().catch(() => {});
    arrancar('video');
  }

  function arrancar(fuente) {
    tracker.reset();
    corriendo = true;
    $('placeholder').style.display = 'none';
    $('btnStop').disabled = false;
    estado('detectando · ' + fuente, 'var(--acc)');
    requestAnimationFrame(bucle);
  }

  function detener() {
    corriendo = false;
    if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
    video.pause();
    $('btnStop').disabled = true;
    if (session) estado('modelo listo', 'var(--ok)');
  }

  /* ---------- Inferencia ---------- */
  function cfgActual(W, H) {
    const orient = $('orient').value;
    const tam = orient === 'h' ? W : H;
    return {
      orient,
      LA: tam * $('lineaA').value / 100,
      LB: tam * $('lineaB').value / 100,
      dist: parseFloat($('distancia').value) || 10,
      limite: parseFloat($('limite').value) || 40,
      conf: parseFloat($('conf').value),
      maxAge: 1.0,
    };
  }

  async function inferir(t) {
    const W = frame.width, H = frame.height;
    const lb = Core.letterbox(W, H, S);
    ictx.fillStyle = 'rgb(114,114,114)';
    ictx.fillRect(0, 0, S, S);
    ictx.drawImage(frame, lb.padX, lb.padY, lb.nw, lb.nh);
    const px = ictx.getImageData(0, 0, S, S).data;
    const plane = S * S;
    for (let i = 0, p = 0; i < plane; i++, p += 4) {
      tensorBuf[i] = px[p] / 255;
      tensorBuf[i + plane] = px[p + 1] / 255;
      tensorBuf[i + 2 * plane] = px[p + 2] / 255;
    }
    const tensor = new ort.Tensor('float32', tensorBuf, [1, 3, S, S]);
    const t0 = performance.now();
    const res = await session.run({ [session.inputNames[0]]: tensor });
    const out = res[session.outputNames[0]];
    const cfg = cfgActual(W, H);
    const dets = Core.decode(out.data, out.dims, nc, cfg.conf, lb);
    const ms = performance.now() - t0;
    fpsEMA = fpsEMA ? fpsEMA * 0.85 + (1000 / ms) * 0.15 : 1000 / ms;
    $('sFps').textContent = fpsEMA.toFixed(1);

    // Separar vehículos y placas
    const idPlaca = Number($('clasePlaca').value);
    const selVeh = $('claseVeh').value;
    const soloPlaca = selVeh !== '*' && Number(selVeh) === idPlaca;
    const placas = dets.filter(d => d.cls === idPlaca);
    const vehs = soloPlaca ? placas
      : dets.filter(d => selVeh === '*' ? d.cls !== idPlaca : d.cls === Number(selVeh));

    const nuevos = tracker.update(vehs, t, cfg);

    // Asociar la mejor placa a cada vehículo
    for (const tr of tracker.tracks) {
      if (tr.lastSeen !== t) continue;
      let pl = null;
      if (soloPlaca) pl = { box: tr.box, score: tr.score };
      else {
        for (const p of placas) {
          const [cx, cy] = Core.centro(p.box);
          if (cx > tr.box[0] && cx < tr.box[2] && cy > tr.box[1] && cy < tr.box[3]
              && (!pl || p.score > pl.score)) pl = p;
        }
      }
      if (pl) {
        const area = (pl.box[2] - pl.box[0]) * (pl.box[3] - pl.box[1]);
        tr.plates = tr.plates || [];
        const menor = tr.plates.length ? Math.min(...tr.plates.map(p => p.area)) : 0;
        if (tr.plates.length < 3 || area > menor * 1.05) {
          tr.plates.push({ area, box: pl.box, img: recorte(pl.box, 0.3) });
          tr.plates.sort((a, b) => b.area - a.area);
          tr.plates.length = Math.min(3, tr.plates.length);
          tr.plate = tr.plates[0];
        }
      }
    }

    for (const tr of nuevos) {
      tr.snap = recorte(soloPlaca ? expandir(tr.box, 2.5, 4, W, H) : tr.box, 0.05);
      if (tr.speed > cfg.limite && !tr.multado) {
        tr.multado = true;
        generarMulta(tr, cfg);
      }
    }
    $('sAutos').textContent = tracker.medidos;
    ultDets = dets;
  }

  function expandir(b, fx, fy, W, H) {
    const [cx, cy] = Core.centro(b), w = (b[2] - b[0]) * fx, h = (b[3] - b[1]) * fy;
    return [Math.max(0, cx - w / 2), Math.max(0, cy - h * 0.75), Math.min(W, cx + w / 2), Math.min(H, cy + h * 0.25)];
  }

  function recorte(b, margen) {
    const w = b[2] - b[0], h = b[3] - b[1];
    const x = Math.max(0, b[0] - w * margen), y = Math.max(0, b[1] - h * margen);
    const ww = Math.min(frame.width - x, w * (1 + 2 * margen)), hh = Math.min(frame.height - y, h * (1 + 2 * margen));
    const c = document.createElement('canvas');
    c.width = Math.max(1, Math.round(ww)); c.height = Math.max(1, Math.round(hh));
    c.getContext('2d').drawImage(frame, x, y, ww, hh, 0, 0, c.width, c.height);
    return c;
  }

  /* ---------- Dibujo ---------- */
  function dibujar() {
    const W = view.width, H = view.height;
    vctx.drawImage(video, 0, 0, W, H);
    const cfg = cfgActual(W, H);
    const k = Math.max(1, W / 640);

    // Líneas A y B
    vctx.lineWidth = 3 * k;
    vctx.setLineDash([12 * k, 8 * k]);
    [['A', cfg.LA, '#facc15'], ['B', cfg.LB, '#38bdf8']].forEach(([n, L, col]) => {
      vctx.strokeStyle = col; vctx.beginPath();
      if (cfg.orient === 'h') { vctx.moveTo(L, 0); vctx.lineTo(L, H); }
      else { vctx.moveTo(0, L); vctx.lineTo(W, L); }
      vctx.stroke();
      vctx.fillStyle = col; vctx.font = `bold ${16 * k}px system-ui`;
      cfg.orient === 'h' ? vctx.fillText(n, L + 4 * k, 20 * k) : vctx.fillText(n, 6 * k, L - 6 * k);
    });
    vctx.setLineDash([]);

    // Todas las detecciones (delgado)
    if ($('verCajas').value === '1') {
      vctx.lineWidth = 1.5 * k;
      for (const d of ultDets) {
        vctx.strokeStyle = 'rgba(255,255,255,.7)';
        vctx.strokeRect(d.box[0], d.box[1], d.box[2] - d.box[0], d.box[3] - d.box[1]);
        vctx.fillStyle = 'rgba(255,255,255,.85)';
        vctx.font = `${11 * k}px system-ui`;
        vctx.fillText(`${nombres[d.cls]} ${(d.score * 100) | 0}%`, d.box[0], d.box[3] + 12 * k);
      }
    }

    // Vehículos rastreados
    for (const tr of tracker.tracks) {
      const b = tr.box;
      let col = '#a3e635', txt = `#${tr.id}`;
      if (tr.speed != null) {
        const exc = tr.speed > cfg.limite;
        col = exc ? '#ef4444' : '#22c55e';
        txt += ` ${tr.speed.toFixed(0)} km/h ${exc ? '⚠' : '✓'}`;
      } else if (tr.tA != null || tr.tB != null) { col = '#facc15'; txt += ' midiendo…'; }
      vctx.strokeStyle = col; vctx.lineWidth = 3 * k;
      vctx.strokeRect(b[0], b[1], b[2] - b[0], b[3] - b[1]);
      vctx.font = `bold ${15 * k}px system-ui`;
      const tw = vctx.measureText(txt).width + 8 * k;
      vctx.fillStyle = col; vctx.fillRect(b[0], b[1] - 20 * k, tw, 20 * k);
      vctx.fillStyle = '#000'; vctx.fillText(txt, b[0] + 4 * k, b[1] - 5 * k);
    }
  }

  /* ---------- Bucle principal ---------- */
  function bucle() {
    if (!corriendo) return;
    if (video.readyState >= 2 && video.videoWidth) {
      const W = video.videoWidth, H = video.videoHeight;
      if (view.width !== W || view.height !== H) { view.width = frame.width = W; view.height = frame.height = H; }
      if (!ocupado && !video.ended && !(video.paused && !stream)) {
        ocupado = true;
        const t = video.currentTime;
        fctx.drawImage(video, 0, 0, W, H);
        inferir(t).catch(e => log('Error inferencia: ' + e.message)).finally(() => { ocupado = false; });
      }
      dibujar();
      if (video.ended) estado('video terminado', 'var(--mut)');
    }
    requestAnimationFrame(bucle);
  }

  /* ---------- Multas ---------- */
  // Pasa un recorte a gris, escala y (opcional) binariza con Otsu; band = [y0,y1] fracción vertical
  function preparar(canvas, alto, band, otsu) {
    const sy = Math.round(canvas.height * band[0]), sh = Math.max(1, Math.round(canvas.height * (band[1] - band[0])));
    const esc = alto / sh;
    const c = document.createElement('canvas');
    c.width = Math.max(1, Math.round(canvas.width * esc)); c.height = Math.round(alto);
    const x = c.getContext('2d');
    x.imageSmoothingQuality = 'high';
    x.drawImage(canvas, 0, sy, canvas.width, sh, 0, 0, c.width, c.height);
    const im = x.getImageData(0, 0, c.width, c.height), d = im.data;
    const g = new Uint8Array(c.width * c.height);
    let mn = 255, mx = 0;
    for (let i = 0, j = 0; i < d.length; i += 4, j++) {
      const v = 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
      g[j] = v; if (v < mn) mn = v; if (v > mx) mx = v;
    }
    const r = Math.max(1, mx - mn);
    for (let j = 0; j < g.length; j++) g[j] = (g[j] - mn) * 255 / r;
    let th = -1;
    if (otsu) {
      const h = new Array(256).fill(0); g.forEach(v => h[v | 0]++);
      let tot = g.length, sum = 0; for (let i = 0; i < 256; i++) sum += i * h[i];
      let sb = 0, wb = 0, mxv = 0; th = 128;
      for (let i = 0; i < 256; i++) {
        wb += h[i]; if (!wb) continue; const wf = tot - wb; if (!wf) break;
        sb += i * h[i]; const mb = sb / wb, mf = (sum - sb) / wf;
        const v = wb * wf * (mb - mf) * (mb - mf); if (v > mxv) { mxv = v; th = i; }
      }
    }
    for (let i = 0, j = 0; i < d.length; i += 4, j++) {
      const v = th >= 0 ? (g[j] > th ? 255 : 0) : g[j];
      d[i] = d[i + 1] = d[i + 2] = v; d[i + 3] = 255;
    }
    x.putImageData(im, 0, 0);
    // borde blanco para que Tesseract no pierda letras pegadas al margen
    const f = document.createElement('canvas'); f.width = c.width + 40; f.height = c.height + 40;
    const fx = f.getContext('2d'); fx.fillStyle = '#fff'; fx.fillRect(0, 0, f.width, f.height); fx.drawImage(c, 20, 20);
    return f;
  }

  /* ---------- OCR con modelo PP-OCRv4 (ONNX) ---------- */
  let recSession = null, recChars = null, recListo = null;
  function cargarRec() {
    if (recListo) return recListo;
    recListo = (async () => {
      recChars = await (await fetch('model/ocr_chars.json')).json();
      recSession = await ort.InferenceSession.create('model/ocr_rec.onnx', { executionProviders: ['wasm'] });
      log('OCR de placas (PP-OCRv4) listo');
    })().catch(e => { log('OCR PP-OCRv4 no disponible: ' + e.message); recSession = null; });
    return recListo;
  }

  // Reconoce una ventana [fx0,fy0,fx1,fy1] (fracciones) de un canvas; devuelve {txt, conf}
  async function reconocerVentana(cv, fx0, fy0, fx1, fy1) {
    const sx = Math.round(cv.width * fx0), sy = Math.round(cv.height * fy0);
    const sw = Math.max(2, Math.round(cv.width * (fx1 - fx0))), sh = Math.max(2, Math.round(cv.height * (fy1 - fy0)));
    const h = 48, w = Math.min(480, Math.max(16, Math.round(sw * h / sh)));
    const c = document.createElement('canvas'); c.width = w; c.height = h;
    const x = c.getContext('2d'); x.imageSmoothingQuality = 'high';
    x.drawImage(cv, sx, sy, sw, sh, 0, 0, w, h);
    const d = x.getImageData(0, 0, w, h).data, buf = new Float32Array(3 * h * w);
    for (let i = 0, p = 0; i < w * h; i++, p += 4) { // el modelo espera BGR normalizado a [-1,1]
      buf[i] = (d[p + 2] / 255 - 0.5) / 0.5;
      buf[w * h + i] = (d[p + 1] / 255 - 0.5) / 0.5;
      buf[2 * w * h + i] = (d[p] / 255 - 0.5) / 0.5;
    }
    const out = await recSession.run({ [recSession.inputNames[0]]: new ort.Tensor('float32', buf, [1, 3, h, w]) });
    const o = out[recSession.outputNames[0]], T = o.dims[1], C = o.dims[2], v = o.data;
    let txt = '', prev = -1, sum = 0, n = 0;
    for (let t = 0; t < T; t++) {
      let bi = 0, bv = -1; for (let k = 0; k < C; k++) { const q = v[t * C + k]; if (q > bv) { bv = q; bi = k; } }
      if (bi !== prev && bi !== 0) { txt += recChars[bi] || ''; sum += bv; n++; }
      prev = bi;
    }
    return { txt: txt.toUpperCase().replace(/\s/g, ''), conf: n ? sum / n : 0 };
  }

  // Barre varias ventanas de cada recorte y vota; es robusto a que el recorte traiga logo/calcomanías/texto extra
  async function leerPlacaRec(lista) {
    await cargarRec();
    if (!recSession) return '';
    const ventanas = [];
    for (const y0 of [0.1, 0.2, 0.3]) for (const y1 of [0.7, 0.8, 0.9]) for (const x0 of [0, 0.15]) for (const x1 of [0.85, 1]) ventanas.push([x0, y0, x1, y1]);
    const acc = {}, forma = {};
    for (const cv of lista) {
      for (const [x0, y0, x1, y1] of ventanas) {
        try {
          const r = await reconocerVentana(cv, x0, y0, x1, y1);
          const k = r.txt.replace(/[^A-Z0-9]/g, '');
          if (k.length < 5 || k.length > 9) continue;
          const bien = /^[A-Z]{2,3}\d{2,4}[A-Z]{0,2}\d{0,3}$/.test(k);
          acc[k] = (acc[k] || 0) + r.conf + (bien ? 2 : 0);
          if (!forma[k] || r.txt.length > forma[k].length) forma[k] = r.txt.replace(/[^A-Z0-9-]/g, '');
        } catch (e) { /* ventana fallida: se ignora */ }
      }
    }
    const ord = Object.entries(acc).sort((a, b) => b[1] - a[1]);
    if (!ord.length) return '';
    log('OCR PP-OCRv4 candidatos: ' + ord.slice(0, 4).map(e => e[0] + ' (' + e[1].toFixed(1) + ')').join(' | '));
    const k = ord[0][0];
    const f = (forma[k] || '').replace(/^-+|-+$/g, '');
    if (f.includes('-') && f.replace(/-/g, '') === k) return f;
    const m = k.match(/^([A-Z]{2,3})(\d{2,4})([A-Z]{0,2}\d{0,3})$/);
    return m ? m[1] + '-' + m[2] + (m[3] ? '-' + m[3] : '') : k;
  }

  // Lee la placa probando varios recortes, variantes de pre-procesado y modos de segmentación
  async function leerPlaca(canvases) {
    const lista = (Array.isArray(canvases) ? canvases : [canvases]).filter(Boolean);
    if (!lista.length) return '';
    try { const r = await leerPlacaRec(lista); if (r) return r; } catch (e) { log('OCR PP-OCRv4 falló: ' + e.message); }
    await prepararOCR();
    if (!ocrWorker) return '';
    const variantes = [
      { band: [0, 1], otsu: false, psm: '7' },
      { band: [0.28, 0.82], otsu: true, psm: '7' },
      { band: [0.28, 0.82], otsu: false, psm: '7' },
      { band: [0.2, 0.9], otsu: true, psm: '8' },
    ];
    const cand = [];
    for (const cv of lista) {
      for (const v of variantes) {
        try {
          await ocrWorker.setParameters({ tessedit_pageseg_mode: v.psm });
          const { data } = await ocrWorker.recognize(preparar(cv, 100, v.band, v.otsu));
          const t = Core.limpiarPlaca(data.text);
          if (t.replace(/-/g, '').length >= 3) cand.push(Core.puntuarPlaca(t, data.confidence));
        } catch (e) { /* variante fallida: se ignora */ }
      }
    }
    if (!cand.length) return '';
    // Votación: se suman los puntajes de lecturas iguales y gana el mayor total
    const acc = {};
    cand.forEach(c => { acc[c.txt] = (acc[c.txt] || 0) + c.score; });
    const mejor = Object.entries(acc).sort((a, b) => b[1] - a[1])[0][0];
    log('OCR candidatos: ' + cand.map(c => c.txt).join(' | ') + ' → ' + mejor);
    return mejor;
  }

  function alerta() {
    const f = $('flash'); f.style.opacity = 1; setTimeout(() => f.style.opacity = 0, 700);
    if (navigator.vibrate) navigator.vibrate([200, 100, 200]);
    try {
      const ac = new (window.AudioContext || window.webkitAudioContext)();
      [0, 0.25].forEach(dt => {
        const o = ac.createOscillator(), g = ac.createGain();
        o.type = 'square'; o.frequency.value = 880;
        g.gain.value = 0.15; o.connect(g); g.connect(ac.destination);
        o.start(ac.currentTime + dt); o.stop(ac.currentTime + dt + 0.18);
      });
    } catch (_) {}
  }

  async function generarMulta(tr, cfg) {
    alerta();
    multas++; folio++;
    $('sMultas').textContent = multas;
    $('sinMultas')?.remove();
    const fecha = new Date();
    const m = {
      folio: `SIM-${fecha.toISOString().slice(0, 10).replace(/-/g, '')}-${String(folio).padStart(4, '0')}`,
      fecha: fecha.toLocaleString('es-MX'),
      vel: tr.speed, lim: cfg.limite, dist: cfg.dist, dt: tr.dt, id: tr.id,
      monto: Core.multaMonto(tr.speed, cfg.limite),
      placa: '', snap: tr.snap, plateImg: tr.plate ? tr.plate.img : null, plateImgs: tr.plates ? tr.plates.map(p => p.img) : [],
    };
    const el = document.createElement('div');
    el.className = 'ticket';
    el.innerHTML = `
      <div class="head"><b>🚨 EXCESO DE VELOCIDAD</b><span>${m.folio}</span></div>
      <div class="body">
        <div class="full"><div class="k">Placa</div>
          <input class="placa-input" value="leyendo…" aria-label="Placa"></div>
        <div><div class="k">Velocidad medida</div><div class="v" style="color:#dc2626">${m.vel.toFixed(1)} km/h</div></div>
        <div><div class="k">Límite</div><div class="v">${m.lim} km/h</div></div>
        <div><div class="k">Exceso</div><div class="v">+${(m.vel - m.lim).toFixed(1)} km/h</div></div>
        <div><div class="k">Monto (simulado)</div><div class="v">$${m.monto.toLocaleString('es-MX')} MXN</div></div>
        <div><div class="k">Fecha y hora</div><div class="v">${m.fecha}</div></div>
        <div><div class="k">Medición</div><div class="v">${m.dist} m en ${m.dt.toFixed(2)} s</div></div>
        <div class="imgs"></div>
      </div>
      <div class="foot"><button class="dl">⬇️ Descargar ticket</button>
        <button class="sh sec">📤 Compartir</button></div>
      <div class="sim">Documento SIMULADO generado por un modelo YOLO con fines académicos
        (Redes Neuronales, Universidad de Sonora). No tiene validez oficial.</div>`;
    const imgs = el.querySelector('.imgs');
    if (m.snap) { const i = new Image(); i.src = m.snap.toDataURL('image/jpeg', 0.85); imgs.appendChild(i); }
    if (m.plateImg) { const i = new Image(); i.className = 'pl'; i.src = m.plateImg.toDataURL(); imgs.appendChild(i); }
    $('tickets').prepend(el);
    const input = el.querySelector('.placa-input');

    m.placa = await leerPlaca(m.plateImgs && m.plateImgs.length ? m.plateImgs : m.plateImg).catch(() => '');
    input.value = m.placa || (m.plateImg ? 'NO LEGIBLE' : 'SIN PLACA');
    log(`Multa ${m.folio}: auto #${m.id} ${m.vel.toFixed(1)} km/h placa=${input.value}`);
    input.addEventListener('input', () => { m.placa = Core.limpiarPlaca(input.value); });

    el.querySelector('.dl').onclick = () => {
      m.placa = Core.limpiarPlaca(input.value) || input.value;
      const a = document.createElement('a');
      a.href = ticketCanvas(m).toDataURL('image/png');
      a.download = `multa_${m.folio}.png`;
      a.click();
    };
    el.querySelector('.sh').onclick = async () => {
      m.placa = Core.limpiarPlaca(input.value) || input.value;
      const blob = await new Promise(r => ticketCanvas(m).toBlob(r, 'image/png'));
      const file = new File([blob], `multa_${m.folio}.png`, { type: 'image/png' });
      if (navigator.canShare && navigator.canShare({ files: [file] })) {
        navigator.share({ files: [file], title: 'Multa simulada ' + m.folio }).catch(() => {});
      } else el.querySelector('.dl').click();
    };
  }

  // Dibuja el ticket como imagen PNG
  function ticketCanvas(m) {
    const Wt = 720, c = document.createElement('canvas');
    const snapH = m.snap ? Math.min(360, Math.round(m.snap.height * (Wt - 40) * 0.6 / m.snap.width)) : 0;
    c.width = Wt; c.height = 560 + snapH;
    const x = c.getContext('2d');
    x.fillStyle = '#fff'; x.fillRect(0, 0, c.width, c.height);
    x.fillStyle = '#dc2626'; x.fillRect(0, 0, Wt, 80);
    x.fillStyle = '#fff'; x.font = 'bold 30px system-ui'; x.fillText('EXCESO DE VELOCIDAD', 24, 50);
    x.font = '18px system-ui'; x.textAlign = 'right'; x.fillText(m.folio, Wt - 24, 50); x.textAlign = 'left';

    x.fillStyle = '#64748b'; x.font = '14px system-ui'; x.fillText('PLACA', 24, 116);
    x.strokeStyle = '#111'; x.lineWidth = 3; x.strokeRect(24, 126, 330, 64);
    x.fillStyle = '#111'; x.font = 'bold 40px ui-monospace, Consolas, monospace';
    x.fillText(m.placa || '—', 38, 172);
    if (m.plateImg) {
      const h = 64, w = Math.min(320, m.plateImg.width * h / m.plateImg.height);
      x.drawImage(m.plateImg, 376, 126, w, h);
    }
    const filas = [
      ['Velocidad medida', `${m.vel.toFixed(1)} km/h`], ['Límite permitido', `${m.lim} km/h`],
      ['Exceso', `+${(m.vel - m.lim).toFixed(1)} km/h`], ['Monto (simulado)', `$${m.monto.toLocaleString('es-MX')} MXN`],
      ['Fecha y hora', m.fecha], ['Medición', `${m.dist} m en ${m.dt.toFixed(2)} s`],
    ];
    filas.forEach(([k, v], i) => {
      const cx = 24 + (i % 2) * 350, cy = 230 + Math.floor(i / 2) * 64;
      x.fillStyle = '#64748b'; x.font = '14px system-ui'; x.fillText(k.toUpperCase(), cx, cy);
      x.fillStyle = i === 0 ? '#dc2626' : '#111'; x.font = 'bold 24px system-ui'; x.fillText(v, cx, cy + 28);
    });
    let y = 430;
    if (m.snap) {
      const w = m.snap.width * snapH / m.snap.height;
      x.drawImage(m.snap, (Wt - w) / 2, y, w, snapH); y += snapH + 16;
    }
    x.fillStyle = '#64748b'; x.font = '13px system-ui';
    x.fillText('Documento SIMULADO generado con YOLO para la materia de Redes Neuronales', 24, y + 30);
    x.fillText('(Universidad de Sonora). Sin validez oficial.', 24, y + 50);
    return c;
  }

  /* ---------- Eventos UI ---------- */
  $('btnCam').onclick = iniciarCamara;
  $('btnStop').onclick = detener;
  $('fileVideo').onchange = e => { const f = e.target.files[0]; if (f && session) iniciarVideo(f); };
  $('rate').onchange = () => { video.playbackRate = parseFloat($('rate').value); };
  $('lineaA').oninput = () => { $('vA').textContent = $('lineaA').value + '%'; };
  $('lineaB').oninput = () => { $('vB').textContent = $('lineaB').value + '%'; };
  $('conf').oninput = () => { $('vConf').textContent = $('conf').value; };
  $('orient').onchange = () => tracker.reset();
  video.addEventListener('seeked', () => tracker.reset());

  cargarModelo();
})();
