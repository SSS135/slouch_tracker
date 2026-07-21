# Slouch model container v1

This is the durable model-pair payload contract. Multi-byte values are little-endian. Integers are unsigned unless named otherwise. Floating values use IEEE-754 and must be finite.

## Container

| Field | Encoding |
|---|---|
| magic | 4 ASCII bytes `SLMD` |
| envelope version | `u16`, exactly `1` |
| classifier-state version | `u16` |
| record count | `u32`, 1–256 |
| records | concatenated in strictly increasing UTF-8 byte order by name |

Each record is:

| Field | Encoding |
|---|---|
| name length/name | `u16` (1–64), then lowercase ASCII bytes `[a-z0-9._-]` |
| kind | `u8`: 1=`u8`, 2=`u32`, 3=`u64`, 4=`i64`, 5=`f32`, 6=`f64`, 7=`utf8`, 8=`bytes`, 9=`tensor_f32`, 10=`tensor_u8`, 11=`tensor_u32` |
| rank | `u8`, 0 for scalar/string/bytes, 1–4 for tensors |
| dimensions | `rank × u32`, each nonzero |
| byte length | `u64` |
| bytes | exact payload bytes |

Reject duplicate/unknown records, noncanonical order, reserved kinds, inconsistent rank/length, trailing bytes, non-finite floats, tensors above 16,777,216 elements, strings/bytes above 1 MiB unless the record is a model tensor, any model tensor above 64 MiB, or a container above 256 MiB. Text is valid UTF-8. Booleans are `u8` 0/1. Optional tensors use a separate `*.present` `u8` record; omitted records are never inferred from zero length.

The SQLite `payload_sha256` is SHA-256 over the complete container from magic through the final record. It is not embedded in the hashed bytes.

## Common required records

Every envelope contains:

- `classifier.id` (`utf8`) and `classifier.state_version` (`u32`)
- `role` (`utf8`: `presence` or `posture`)
- `feature.ids` (`utf8`, comma-separated in registry order) and `feature.input_dimension` (`u32`)
- `trained_at_ms` (`i64`), `dataset.version` (`u64`), `training_config.sha256` (`bytes`, exactly 32). Both role envelopes in one generation contain the same pair-level fingerprint.
- `normalization.mode` (`utf8`: `none`, `layer`, `z_score`, or `calibrated`)
- `reduction.method` (`utf8`: `none`, `random_projection`, or `pca`) and `reduction.output_dimension` (`u32`)
- optional cross-validation records use this exact allowlist and no others: `metric.cv_accuracy`, `metric.cv_std`, `metric.mcc`, and `metric.f1_score` as `f64`; `metric.confusion_matrix` as `tensor_u32 [2,2]`; `metric.fold_accuracies` as `tensor_f32 [folds]`; and `metric.cv_type` as `utf8` (`temporal_block` or `shuffled_stratified`). All seven records are present when cross-validation ran and all are absent otherwise.

For `z_score` and `calibrated` normalization require `normalization.mean` and `normalization.std` as `tensor_f32 [input_dimension]`; every std is positive. `calibrated` reuses these exact records (mean/std fit on the reference class only); it adds no new record type. Layer/none have no tensors.

Random projection requires `reduction.matrix` `tensor_f32 [output,input]`, `reduction.rng` (`utf8`), and `reduction.seed` (`utf8`, exact seed bytes). PCA requires `reduction.mean` `tensor_f32 [input]`, `reduction.components` `tensor_f32 [output,input]`, and `reduction.explained_variance` `tensor_f32 [output]`. None has no reduction tensors and output equals input.

## Classifier records

| Classifier | State version | Required classifier-specific records |
|---|---:|---|
| `mlp` | 1 | `mlp.layer_shapes` `tensor_u32 [layers+1]`; for each layer `i`, `mlp.i.weights` `tensor_f32 [out,in]` and `mlp.i.biases` `tensor_f32 [out]`; `mlp.hidden_layers`/`mlp.hidden_size` `u32`; `mlp.class_weights` `tensor_f32 [2]` |
| `knn` | 1 | `knn.training_data` `tensor_f32 [samples,dimension]`; `knn.training_labels` `tensor_u8 [samples]` values 0/1; `knn.k` `u32` in 1..=samples; `knn.kernel` `utf8` (`cosine` or `rbf`); `knn.gamma` `f64`, positive for RBF |
| `svm` | 1 | `svm.weights` `tensor_f32 [dimension]`; `svm.bias` `f32`; `svm.class_weights` `tensor_f32 [2]` |
| `kmeans_prototype` | 1 | `kmp.centroids` `tensor_f32 [clusters,dimension]`; `kmp.prototype_present` `tensor_u8 [clusters,2]`; `kmp.prototypes` `tensor_f32 [clusters,2,dimension]`; `kmp.global_good`/`kmp.global_bad` `tensor_f32 [dimension]`; `kmp.temperature` `f64` |
| `gaussian_nb` | 1 | `gnb.means`/`gnb.variances` `tensor_f32 [2,dimension]`; `gnb.priors` `tensor_f32 [2]`; `gnb.epsilon` `f32`; variances/priors positive and priors sum within `1e-6` of one |
| `kmeans_logistic` | 1 | `kml.centroids` `tensor_f32 [clusters,dimension]`; `kml.model_present` `tensor_u8 [clusters]`; a canonical MLP record set prefixed `kml.cluster.N.` for each present model; canonical MLP records prefixed `kml.global.`; `kml.temperature` `f64` |

Classifier payloads must contain exactly their common records, normalization/reduction records, and the selected classifier allowlist. Missing and extra records are errors.

## Training-config fingerprint

`training_config.sha256` is SHA-256 of a second canonical record container with magic `SLCF`, version `1`, the same record framing/order rules, and no model tensors. It contains the complete model-pair training configuration. Every record is namespaced `presence.` or `posture.` and each namespace contains exactly:

- ordered feature IDs and registry dimensions;
- normalization mode;
- reduction method/components and reduction RNG/seed when applicable;
- classifier ID and every effective hyperparameter after defaults;
- cross-validation enabled/folds/seed;
- class-weight setting;
- probability threshold.

Pair-wide dataset selection/snapshot options use the `pair.` prefix. Numeric hyperparameters use `f64`, counts use `u32`, booleans use `u8`, and strings use exact lowercase registry identifiers. Defaults are materialized before hashing. The resulting one pair-level hash plus dataset version binds both envelopes in a model generation; role-specific hashes are forbidden.
