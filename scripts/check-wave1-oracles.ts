import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import * as tf from '@tensorflow/tfjs';
import { GaussianNBClassifier } from '../src/services/ml/naiveBayesClassifier';
import { poolFeaturesMean, poolFeaturesStd } from '../src/services/ml/rtmposeFeatures';

const poolingFixture = JSON.parse(
  readFileSync('src-tauri/fixtures/math/rtmpose-pooling-v1.json', 'utf8'),
);
const gaussianFixture = JSON.parse(
  readFileSync('src-tauri/fixtures/classifiers/gaussian-nb-v1.json', 'utf8'),
);

await tf.setBackend('cpu');
await tf.ready();

const cancellation = poolingFixture.backbone.cancellationSensitive;
const lane = new Float32Array(cancellation.length).fill(cancellation.middleValue);
lane[0] = cancellation.first;
lane[lane.length - 1] = cancellation.last;
const mean = poolFeaturesMean(lane, [1, 1, 1, lane.length], [2, 3])[0];
const std = poolFeaturesStd(lane, [1, 1, 1, lane.length], [2, 3])[0];
assert.equal(mean, cancellation.mean);
assert.equal(std, cancellation.std);

const highDimensional = gaussianFixture.highDimensionalCase;
const features: Float32Array[] = [];
const labels: number[] = [];
for (let index = 0; index < highDimensional.samplesPerClass; index += 1) {
  features.push(
    new Float32Array(highDimensional.dimensions).fill(
      highDimensional.class0Alternating[index % 2],
    ),
  );
  labels.push(0);
}
for (let index = 0; index < highDimensional.samplesPerClass; index += 1) {
  features.push(
    new Float32Array(highDimensional.dimensions).fill(
      highDimensional.class1Alternating[index % 2],
    ),
  );
  labels.push(1);
}
const classifier = new GaussianNBClassifier({ params: {} });
classifier.train(features, new Int32Array(labels));
const probabilityGood = classifier.predictProba(
  new Float32Array(highDimensional.dimensions).fill(highDimensional.probeValue),
);
assert.equal(probabilityGood, highDimensional.probabilityGood);

console.log(JSON.stringify({ backend: tf.getBackend(), mean, std, probabilityGood }));
