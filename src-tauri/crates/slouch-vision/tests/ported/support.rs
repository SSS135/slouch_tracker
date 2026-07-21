use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use ndarray::Array4;
use slouch_domain::ported::messages::schemas::{
    DimensionalityReductionConfig, DimensionalityReductionMethod, ImageData, NormalizationMode,
    SerializedClassifier, SerializedClassifierState, SerializedFeatureExtractor,
    SerializedGaussianNb, SerializedModel,
};
use slouch_vision::ported::inference_worker::{
    ClassificationInput, ClassifierModel, InferenceRuntime, InferenceSession, ModelFactory,
    SessionOptions, SessionOutput, SessionOutputMap, WorkerError, WorkerLogger,
};

pub const CLS_P5: &str = "/bbox_head/cls_convs.2.1/pointwise_conv/activate/Mul_output_0";
pub const REG_P5: &str = "/bbox_head/reg_convs.2.1/pointwise_conv/activate/Mul_output_0";

#[derive(Clone, Debug, Default, PartialEq)]
pub struct RuntimeTrace {
    pub creates: Vec<(String, String, SessionOptions)>,
    pub waits: Vec<Duration>,
    pub runs: usize,
}

pub enum CreateOutcome {
    Session(VecDeque<Result<SessionOutputMap, WorkerError>>),
    Error(String),
}

pub struct TestRuntime {
    pub outcomes: VecDeque<CreateOutcome>,
    pub trace: Arc<Mutex<RuntimeTrace>>,
}

impl TestRuntime {
    pub fn new(
        outcomes: impl IntoIterator<Item = CreateOutcome>,
    ) -> (Self, Arc<Mutex<RuntimeTrace>>) {
        let trace = Arc::new(Mutex::new(RuntimeTrace::default()));
        (
            Self {
                outcomes: outcomes.into_iter().collect(),
                trace: Arc::clone(&trace),
            },
            trace,
        )
    }
}

struct TestSession {
    runs: VecDeque<Result<SessionOutputMap, WorkerError>>,
    trace: Arc<Mutex<RuntimeTrace>>,
}

impl InferenceSession for TestSession {
    fn run(&mut self, _input: &Array4<f32>) -> Result<SessionOutputMap, WorkerError> {
        self.trace.lock().expect("trace").runs += 1;
        self.runs
            .pop_front()
            .unwrap_or_else(|| Err(WorkerError::Inference("no injected session output".into())))
    }
}

impl InferenceRuntime for TestRuntime {
    fn create_session(
        &mut self,
        path: &str,
        model_name: &str,
        options: SessionOptions,
    ) -> Result<Box<dyn InferenceSession>, WorkerError> {
        self.trace.lock().expect("trace").creates.push((
            path.to_owned(),
            model_name.to_owned(),
            options,
        ));
        match self.outcomes.pop_front() {
            Some(CreateOutcome::Session(runs)) => Ok(Box::new(TestSession {
                runs,
                trace: Arc::clone(&self.trace),
            })),
            Some(CreateOutcome::Error(error)) => Err(WorkerError::Model(error)),
            None => Err(WorkerError::Model("no injected create outcome".into())),
        }
    }

    fn wait_before_retry(&mut self, delay: Duration) {
        self.trace.lock().expect("trace").waits.push(delay);
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ModelStats {
    pub predicts: usize,
    pub disposes: usize,
}

#[derive(Clone, Default)]
pub struct TestFactory {
    pub stats: Arc<Mutex<HashMap<i64, ModelStats>>>,
    pub trace: Arc<Mutex<Vec<String>>>,
    pub load_failures: Arc<Mutex<HashSet<i64>>>,
    pub predict_failures: Arc<Mutex<HashSet<i64>>>,
}

struct TestModel {
    key: i64,
    probability: f64,
    stats: Arc<Mutex<HashMap<i64, ModelStats>>>,
    trace: Arc<Mutex<Vec<String>>>,
    predict_failures: Arc<Mutex<HashSet<i64>>>,
}

impl ClassifierModel for TestModel {
    fn predict(&mut self, _input: &ClassificationInput<'_>) -> Result<f64, String> {
        self.stats
            .lock()
            .expect("stats")
            .entry(self.key)
            .or_default()
            .predicts += 1;
        self.trace
            .lock()
            .expect("factory trace")
            .push(format!("predict {}", self.key));
        if self
            .predict_failures
            .lock()
            .expect("failures")
            .contains(&self.key)
        {
            Err(format!("prediction {} failed", self.key))
        } else {
            Ok(self.probability)
        }
    }

    fn dispose(&mut self) {
        self.stats
            .lock()
            .expect("stats")
            .entry(self.key)
            .or_default()
            .disposes += 1;
        self.trace
            .lock()
            .expect("factory trace")
            .push(format!("dispose {}", self.key));
    }
}

impl ModelFactory for TestFactory {
    fn load(&self, serialized: SerializedModel) -> Result<Box<dyn ClassifierModel>, String> {
        let key = (serialized.trained_at * 100.0).round() as i64;
        self.trace
            .lock()
            .expect("factory trace")
            .push(format!("load {key}"));
        if self.load_failures.lock().expect("failures").contains(&key) {
            return Err(format!("injected load failure {key}"));
        }
        self.stats.lock().expect("stats").entry(key).or_default();
        Ok(Box::new(TestModel {
            key,
            probability: serialized.trained_at,
            stats: Arc::clone(&self.stats),
            trace: Arc::clone(&self.trace),
            predict_failures: Arc::clone(&self.predict_failures),
        }))
    }
}

#[derive(Clone, Default)]
pub struct TestLogger(pub Arc<Mutex<Vec<(String, String)>>>);

impl WorkerLogger for TestLogger {
    fn debug(&self, message: &str) {
        self.0
            .lock()
            .expect("logs")
            .push(("debug".into(), message.into()));
    }
    fn info(&self, message: &str) {
        self.0
            .lock()
            .expect("logs")
            .push(("info".into(), message.into()));
    }
    fn warn(&self, message: &str) {
        self.0
            .lock()
            .expect("logs")
            .push(("warn".into(), message.into()));
    }
    fn error(&self, message: &str) {
        self.0
            .lock()
            .expect("logs")
            .push(("error".into(), message.into()));
    }
}

pub fn model(probability: f64) -> SerializedModel {
    SerializedModel {
        feature_extractor: SerializedFeatureExtractor {
            feature_types: vec!["gau_features".into()],
            normalization_mode: NormalizationMode::None,
            dim_reduction_config: DimensionalityReductionConfig {
                method: DimensionalityReductionMethod::None,
                components: 1,
            },
            concatenated_dimensions: 1,
            normalization_mean: None,
            normalization_std: None,
            dim_reduction_transformer: None,
        },
        classifier: SerializedClassifier {
            classifier_id: "gaussian_nb".into(),
            state: SerializedClassifierState::GaussianNb(SerializedGaussianNb {
                class_means: [vec![0.0], vec![1.0]],
                class_variances: [vec![1.0], vec![1.0]],
                class_priors: [0.5, 0.5],
                epsilon: 1e-9,
            }),
        },
        trained_at: probability,
        version: 1.0,
    }
}

pub fn image(width: u32, height: u32) -> ImageData {
    ImageData {
        data: vec![127; width as usize * height as usize * 4],
        width,
        height,
    }
}

pub fn detector_outputs(dets: Vec<f32>, labels: Vec<i64>, feature_value: f32) -> SessionOutputMap {
    HashMap::from([
        ("dets".into(), SessionOutput::F32(dets)),
        ("labels".into(), SessionOutput::I64(labels)),
        (
            CLS_P5.into(),
            SessionOutput::F32(vec![feature_value; 64 * 10 * 10]),
        ),
        (
            REG_P5.into(),
            SessionOutput::F32(vec![feature_value * 2.0; 64 * 10 * 10]),
        ),
    ])
}

pub fn pose_outputs() -> SessionOutputMap {
    let mut x = vec![0.0; 17 * 384];
    let mut y = vec![0.0; 17 * 512];
    for keypoint in 0..17 {
        x[keypoint * 384 + 100 + keypoint] = 0.8;
        y[keypoint * 512 + 120 + keypoint] = 0.6;
    }
    HashMap::from([
        ("simcc_x".into(), SessionOutput::F32(x)),
        ("simcc_y".into(), SessionOutput::F32(y)),
        (
            "backbone_features".into(),
            SessionOutput::F32(vec![0.25; 768 * 8 * 6]),
        ),
        (
            "gau_features".into(),
            SessionOutput::F32(vec![0.5; 17 * 256]),
        ),
    ])
}
