import * as ort from 'onnxruntime-web/wasm';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { bits, envelope, isMain, parseWriteFlag, root, sha256, sha256File, source, writeOrCheck } from './common';

const generator = 'scripts/oracles/generate-vision-inference.ts';
const wasmPath = 'node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm';
const clsName = '/bbox_head/cls_convs.2.1/pointwise_conv/activate/Mul_output_0';
const regName = '/bbox_head/reg_convs.2.1/pointwise_conv/activate/Mul_output_0';
type Frame = { id: string; width: number; height: number; rgba: Uint8ClampedArray };

if (typeof globalThis.ImageData === 'undefined') {
  class NodeImageData {
    readonly data: Uint8ClampedArray;
    readonly width: number;
    readonly height: number;
    readonly colorSpace = 'srgb';
    constructor(data: Uint8ClampedArray, width: number, height: number) {
      this.data = data;
      this.width = width;
      this.height = height;
    }
  }
  Object.defineProperty(globalThis, 'ImageData', { value: NodeImageData, configurable: true });
}

function imageData(frame: Frame): ImageData {
  return new ImageData(frame.rgba, frame.width, frame.height);
}

function makeFrame(id: string, width: number, height: number, kind: string): Frame {
  const rgba = new Uint8ClampedArray(width * height * 4); let state = 0x12345678;
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const index = (y * width + x) * 4; let r = 236, g = 238, b = 242;
    if (kind === 'gradient') { r = x * 255 / Math.max(1, width - 1); g = y * 255 / Math.max(1, height - 1); b = (x + y) % 256; }
    if (kind === 'noise') { state = (Math.imul(state, 1664525) + 1013904223) >>> 0; r = state & 255; g = state >>> 8 & 255; b = state >>> 16 & 255; }
    rgba[index] = r; rgba[index + 1] = g; rgba[index + 2] = b; rgba[index + 3] = 255;
  }
  const disk = (cx: number, cy: number, radius: number, color: [number, number, number]) => { for (let y = Math.max(0, cy - radius); y < Math.min(height, cy + radius); y++) for (let x = Math.max(0, cx - radius); x < Math.min(width, cx + radius); x++) if ((x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2) { const i = (y * width + x) * 4; rgba.set([...color, 255], i); } };
  const rect = (x1: number, y1: number, x2: number, y2: number, color: [number, number, number]) => { for (let y = Math.max(0, y1); y < Math.min(height, y2); y++) for (let x = Math.max(0, x1); x < Math.min(width, x2); x++) { const i = (y * width + x) * 4; rgba.set([...color, 255], i); } };
  const person = (cx: number, top: number, scale: number) => { const color: [number, number, number] = [45, 58, 78]; disk(cx, top + Math.round(10 * scale), Math.round(9 * scale), [184, 128, 96]); rect(cx - Math.round(8 * scale), top + Math.round(19 * scale), cx + Math.round(8 * scale), top + Math.round(55 * scale), color); rect(cx - Math.round(22 * scale), top + Math.round(23 * scale), cx + Math.round(22 * scale), top + Math.round(30 * scale), color); rect(cx - Math.round(8 * scale), top + Math.round(55 * scale), cx - Math.round(1 * scale), top + Math.round(88 * scale), color); rect(cx + Math.round(1 * scale), top + Math.round(55 * scale), cx + Math.round(8 * scale), top + Math.round(88 * scale), color); };
  if (kind === 'person') person(Math.floor(width / 2), Math.floor(height * 0.05), Math.min(width, height) / 110);
  if (kind === 'two-person') { person(Math.floor(width * 0.3), Math.floor(height * 0.08), Math.min(width, height) / 135); person(Math.floor(width * 0.7), Math.floor(height * 0.03), Math.min(width, height) / 110); }
  if (kind === 'edge-person') person(14, Math.floor(height * 0.04), Math.min(width, height) / 90);
  if (kind === 'crop-person') person(Math.floor(width / 2), 0, Math.min(width, height) / 80);
  return { id, width, height, rgba };
}

const f32Sha = (values: Float32Array) => sha256(new Uint8Array(values.buffer, values.byteOffset, values.byteLength));
const summary = (values: Float32Array, includeValues = false) => ({
  length: values.length,
  shape: [],
  sha256: f32Sha(values),
  first16: Array.from(values.slice(0, 16)),
  first16Bits: bits(values.slice(0, 16)),
  ...(includeValues ? { values: Array.from(values) } : {}),
});

export async function generateVisionInference(write: boolean): Promise<void> {
  const {
    compatibilityProcessFrameWithSessions,
    cropImageData,
    decodeSimCC,
    expandBbox,
    preprocessRTMDet,
    preprocessRTMW3D,
    selectPersonBbox,
    shouldRunPostureForPresence,
  } = await import('../../src/workers/inference-worker');
  const frames = [makeFrame('empty-landscape', 160, 96, 'empty'), makeFrame('noise-portrait', 96, 160, 'noise'), makeFrame('single-silhouette-square', 128, 128, 'person'), makeFrame('multiple-silhouettes', 192, 128, 'two-person'), makeFrame('edge-clipped-silhouette', 128, 128, 'edge-person'), makeFrame('boundary-crop-silhouette', 127, 97, 'crop-person'), makeFrame('odd-gradient-boundary', 97, 95, 'gradient')];
  for (const frame of frames) writeOrCheck(`src-tauri/fixtures/vision/frames/${frame.id}.rgba`, Buffer.from(frame.rgba), write);
  ort.env.wasm.numThreads = 1; ort.env.wasm.wasmPaths = pathToFileURL(resolve(root, 'node_modules/onnxruntime-web/dist') + '/').href;
  const settings = { executionProviders: ['wasm'] as const, graphOptimizationLevel: 'all' as const };
  const detSession = await ort.InferenceSession.create(new Uint8Array(readFileSync(resolve(root, 'public/rtmdet-nano.onnx'))), settings);
  const poseSession = await ort.InferenceSession.create(new Uint8Array(readFileSync(resolve(root, 'public/rtmpose-m.onnx'))), settings);
  const cases = [];
  for (const frame of frames) {
    const input = imageData(frame);
    const production = await compatibilityProcessFrameWithSessions(input, detSession, poseSession);
    const detPrep = preprocessRTMDet(input);
    const scaledW = Math.round(frame.width * detPrep.scale), scaledH = Math.round(frame.height * detPrep.scale);
    const detOutputs = await detSession.run({ input: new ort.Tensor('float32', detPrep.tensor, [1, 3, 320, 320]) });
    const dets = detOutputs.dets.data as Float32Array, labels = detOutputs.labels.data as BigInt64Array;
    const cls = detOutputs[clsName].data as Float32Array, reg = detOutputs[regName].data as Float32Array;
    const selection = selectPersonBbox(dets, labels, detPrep, frame.width, frame.height);
    if (selection.personFound !== production.personFound) throw new Error(`${frame.id}: production and diagnostic detector routes disagree`);
    let rtmpose = null;
    let pipeline: Record<string, unknown> = { poseRuns: 0, bbox: null, keypoints: null };
    if (selection.selected) {
      const expanded = expandBbox(selection.selected, 0.2, frame.width, frame.height);
      const cropped = cropImageData(input, expanded.expanded);
      const posePrep = preprocessRTMW3D(cropped);
      const poseOutputs = await poseSession.run({ input: new ort.Tensor('float32', posePrep, [1, 3, 256, 192]) });
      const simccX = poseOutputs.simcc_x.data as Float32Array, simccY = poseOutputs.simcc_y.data as Float32Array;
      const decodedKeypoints = decodeSimCC(simccX, simccY);
      if (!production.bbox || !production.keypoints) throw new Error(`${frame.id}: production pose route omitted bbox or keypoints`);
      pipeline = { poseRuns: 1, bbox: production.bbox, keypoints: production.keypoints, crop: { width: cropped.width, height: cropped.height } };
      rtmpose = { preprocessing: { tensor: summary(posePrep) }, outputShapes: { simccX: poseOutputs.simcc_x.dims, simccY: poseOutputs.simcc_y.dims, backbone: poseOutputs.backbone_features.dims, gau: poseOutputs.gau_features.dims }, decodedKeypoints, transformedKeypoints: production.keypoints, backboneAvg: summary(production.features.backbone_features, true), backboneMax: summary(production.features.backbone_features_max, true), backboneStd: summary(production.features.backbone_features_std, true), gauAvg: summary(production.features.gau_features, true), gauMax: summary(production.features.gau_features_max, true), gauStd: summary(production.features.gau_features_std, true) };
    }
    cases.push({ id: frame.id, frame: { path: `src-tauri/fixtures/vision/frames/${frame.id}.rgba`, width: frame.width, height: frame.height, bytes: frame.rgba.length, sha256: sha256(frame.rgba) },
      preprocessing: { rtmdet: { scale: detPrep.scale, scaledW, scaledH, padW: detPrep.padW, padH: detPrep.padH, tensor: summary(detPrep.tensor) } },
      rtmdet: { outputShapes: { dets: detOutputs.dets.dims, labels: detOutputs.labels.dims, clsP5: detOutputs[clsName].dims, regP5: detOutputs[regName].dims }, dets: summary(dets), labelsSha256: sha256(new Uint8Array(labels.buffer, labels.byteOffset, labels.byteLength)), selection, pooled: summary(production.features.rtmdet_extracted, true) },
      pipeline, rtmpose });
  }
  await detSession.release(); await poseSession.release();

  const detectorRows = [
    { id: 'detector-score-exact-threshold', dets: [10, 10, 30, 30, 0.3], labels: [0] },
    { id: 'bbox-largest-area-tie', dets: [10, 10, 30, 30, 0.4, 40, 40, 60, 60, 0.5], labels: [0, 0] },
  ].map((definition) => ({
    ...definition,
    selection: selectPersonBbox(new Float32Array(definition.dets), BigInt64Array.from(definition.labels.map(BigInt)), { scale: 1, padW: 0, padH: 0 }, 100, 100),
  }));
  const simccX = new Float32Array(17 * 384).fill(-1);
  const simccY = new Float32Array(17 * 512).fill(-1);
  for (let keypoint = 0; keypoint < 17; keypoint++) {
    simccX[keypoint * 384 + 2] = 1; simccX[keypoint * 384 + 3] = 1;
    simccY[keypoint * 512 + 4] = 1; simccY[keypoint * 512 + 5] = 1;
  }
  const postprocessingCases = [
    ...detectorRows,
    { id: 'simcc-first-index-tie', keypoints: decodeSimCC(simccX, simccY) },
  ];
  const cascadeCases = [0.499999, 0.5, 0.500001].map((presentProbability) => ({
    id: `presence-${presentProbability}`,
    presentProbability,
    postureRuns: shouldRunPostureForPresence(presentProbability),
  }));
  writeOrCheck('src-tauri/fixtures/vision/vision-inference-v1.json', envelope('vision/vision-inference-v1', generator, [
    'src/workers/inference-worker.ts', 'src/services/ml/constants.ts', 'src/services/ml/rtmdetFeatures.ts', 'src/services/ml/rtmposeFeatures.ts', 'src/services/image/cropUtils.ts',
  ], 'ORT-Web 1.23.0 WASM EP', cases, {
    tolerances: { vision: { absolute: 0.0002, relative: 0.0002 }, exact: ['ids', 'labels', 'shapes', 'routing', 'poseRunCount', 'decisionsAwayFromThreshold'] },
    corpus: { generator: source(generator), license: 'MIT; generated entirely by repository code', licenseFile: 'LICENSE', privacy: 'synthetic pixels only; no people, biometrics, camera captures, or personal data', categories: ['human-like silhouettes', 'empty', 'noise', 'crop', 'edge', 'odd-boundary'] },
    ortWeb: { package: 'onnxruntime-web', version: '1.23.0', npmIntegrity: 'sha512-w0bvC2RwDxphOUFF8jFGZ/dYw+duaX20jM6V4BIZJPCfK4QuCpB/pVREV+hjYbT3x4hyfa2ZbTaWx4e1Vot0fQ==', artifact: { filename: 'ort-wasm-simd-threaded.wasm', bytes: readFileSync(resolve(root, wasmPath)).length, sha256: sha256File(wasmPath), source: 'lock-resolved npm package; production CDN baseline https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.0/dist/' }, executionProvider: 'wasm', threads: 1, graphOptimization: 'all' },
    models: [source('public/rtmdet-nano.onnx'), source('public/rtmpose-m.onnx')],
    exactBehavior: { detectionThreshold: { operator: '>', value: 0.3 }, bboxSelection: 'largest positive area; first wins ties', simccTies: 'first index wins', bboxExpansion: 0.2, presenceThreshold: { operator: '>=', value: 0.5 } },
    postprocessingCases,
    cascadeCases,
  }), write);
}

if (isMain(import.meta.url)) await generateVisionInference(parseWriteFlag());
