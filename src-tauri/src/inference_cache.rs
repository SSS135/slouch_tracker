//! One-time inference token cache extracted from `actors.rs`.
//!
//! This file is compiled twice on purpose: `actors.rs` includes it as a
//! private submodule for the inference actor loop, and `tests/ipc_security.rs`
//! includes it at the test-crate root so token consumption, eviction, and
//! expiry semantics are pinned adversarially without booting the actor.

use std::{
    collections::{HashMap, VecDeque},
    time::{Duration, Instant},
};

use crate::errors::ApiError;
use slouch_vision::ported::inference_worker::NativeInferenceResult;

pub(crate) const INFERENCE_CACHE_CAPACITY: usize = 32;
pub(crate) const INFERENCE_CACHE_MAX_BYTES: usize = 64 * 1024 * 1024;
pub(crate) const INFERENCE_CACHE_TTL: Duration = Duration::from_secs(120);
pub(crate) const TOMBSTONE_CAPACITY: usize = 64;
pub(crate) const TOMBSTONE_TTL: Duration = Duration::from_secs(300);
pub(crate) const MAX_SAFE_JS_INTEGER: u64 = 9_007_199_254_740_991;

struct CachedInference {
    request_id: u64,
    result: NativeInferenceResult,
    retained_bytes: usize,
    inserted_at: Instant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TombstoneReason {
    Expired,
    Evicted,
    Consumed,
}

struct Tombstone {
    token: u64,
    reason: TombstoneReason,
    inserted_at: Instant,
}

pub(crate) struct InferenceCache {
    entries: HashMap<u64, CachedInference>,
    order: VecDeque<u64>,
    reserved: HashMap<u64, (u64, usize)>,
    tombstones: VecDeque<Tombstone>,
    pub(crate) retained_bytes: usize,
    token_state: u64,
}

impl InferenceCache {
    pub(crate) fn new() -> Self {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0x6a09_e667_f3bc_c909, |duration| {
                duration.as_nanos() as u64 ^ 0xa076_1d64_78bd_642f
            });
        Self::with_seed(seed)
    }

    pub(crate) fn with_seed(seed: u64) -> Self {
        Self {
            entries: HashMap::new(),
            order: VecDeque::new(),
            reserved: HashMap::new(),
            tombstones: VecDeque::new(),
            retained_bytes: 0,
            token_state: seed,
        }
    }

    fn next_token(&mut self, request_id: u64) -> u64 {
        loop {
            self.token_state = self.token_state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut value = self.token_state;
            value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            let token = (value ^ (value >> 31)) & MAX_SAFE_JS_INTEGER;
            if token != 0
                && token != request_id
                && !self.entries.contains_key(&token)
                && !self.reserved.contains_key(&token)
                && !self.tombstones.iter().any(|item| item.token == token)
            {
                return token;
            }
        }
    }

    fn prune(&mut self) {
        let now = Instant::now();
        let expired = self
            .entries
            .iter()
            .filter_map(|(token, entry)| {
                (now.duration_since(entry.inserted_at) > INFERENCE_CACHE_TTL).then_some(*token)
            })
            .collect::<Vec<_>>();
        for token in expired {
            self.remove_entry(token);
            self.remember(token, TombstoneReason::Expired);
        }
        while self
            .tombstones
            .front()
            .is_some_and(|item| now.duration_since(item.inserted_at) > TOMBSTONE_TTL)
        {
            self.tombstones.pop_front();
        }
    }

    pub(crate) fn insert(
        &mut self,
        request_id: u64,
        result: NativeInferenceResult,
    ) -> Result<u64, ApiError> {
        self.prune();
        let retained_bytes = retained_inference_bytes(&result)?;
        if retained_bytes > INFERENCE_CACHE_MAX_BYTES {
            return Err(ApiError::InvalidRequest(
                "inference feature bundle exceeds the 64 MiB token limit".into(),
            ));
        }
        while self.entries.len() + self.reserved.len() >= INFERENCE_CACHE_CAPACITY
            || self
                .retained_bytes
                .checked_add(retained_bytes)
                .is_none_or(|total| total > INFERENCE_CACHE_MAX_BYTES)
        {
            let Some(evicted) = self.order.pop_front() else {
                return Err(ApiError::Busy(
                    "inference token cache is fully reserved".into(),
                ));
            };
            self.remove_entry(evicted);
            self.remember(evicted, TombstoneReason::Evicted);
        }
        let token = self.next_token(request_id);
        self.retained_bytes += retained_bytes;
        self.order.push_back(token);
        self.entries.insert(
            token,
            CachedInference {
                request_id,
                result,
                retained_bytes,
                inserted_at: Instant::now(),
            },
        );
        Ok(token)
    }

    fn remove_entry(&mut self, token: u64) {
        if let Some(entry) = self.entries.remove(&token) {
            self.retained_bytes = self.retained_bytes.saturating_sub(entry.retained_bytes);
        }
        self.order.retain(|value| *value != token);
    }

    fn remember(&mut self, token: u64, reason: TombstoneReason) {
        self.tombstones.push_back(Tombstone {
            token,
            reason,
            inserted_at: Instant::now(),
        });
        while self.tombstones.len() > TOMBSTONE_CAPACITY {
            self.tombstones.pop_front();
        }
    }

    pub(crate) fn checkout(
        &mut self,
        token: u64,
        request_id: u64,
    ) -> Result<NativeInferenceResult, ApiError> {
        self.prune();
        if let Some(entry) = self.entries.remove(&token) {
            if entry.request_id != request_id {
                self.entries.insert(token, entry);
                return Err(ApiError::InvalidRequest(
                    "inference token does not match request ID".into(),
                ));
            }
            self.order.retain(|value| *value != token);
            self.reserved
                .insert(token, (request_id, entry.retained_bytes));
            return Ok(entry.result);
        }
        if self.reserved.contains_key(&token) {
            return Err(ApiError::Busy("inference token is being saved".into()));
        }
        if let Some(tombstone) = self.tombstones.iter().find(|item| item.token == token) {
            let message = match tombstone.reason {
                TombstoneReason::Expired => "inference token expired",
                TombstoneReason::Evicted => "inference token was evicted",
                TombstoneReason::Consumed => "inference token was already consumed",
            };
            return Err(ApiError::InvalidRequest(message.into()));
        }
        Err(ApiError::InvalidRequest("unknown inference token".into()))
    }

    pub(crate) fn restore(
        &mut self,
        token: u64,
        request_id: u64,
        result: NativeInferenceResult,
    ) -> Result<(), ApiError> {
        let Some((reserved_request, retained_bytes)) = self.reserved.remove(&token) else {
            return Err(ApiError::Internal(
                "inference token was not reserved".into(),
            ));
        };
        if reserved_request != request_id {
            self.reserved
                .insert(token, (reserved_request, retained_bytes));
            return Err(ApiError::InvalidRequest(
                "inference token does not match request ID".into(),
            ));
        }
        self.order.push_back(token);
        self.entries.insert(
            token,
            CachedInference {
                request_id,
                result,
                retained_bytes,
                inserted_at: Instant::now(),
            },
        );
        Ok(())
    }

    pub(crate) fn commit(&mut self, token: u64, request_id: u64) -> Result<(), ApiError> {
        let Some((reserved_request, retained_bytes)) = self.reserved.remove(&token) else {
            return Err(ApiError::Internal(
                "inference token was not reserved".into(),
            ));
        };
        if reserved_request != request_id {
            self.reserved
                .insert(token, (reserved_request, retained_bytes));
            return Err(ApiError::InvalidRequest(
                "inference token does not match request ID".into(),
            ));
        }
        self.retained_bytes = self.retained_bytes.saturating_sub(retained_bytes);
        self.remember(token, TombstoneReason::Consumed);
        Ok(())
    }

    pub(crate) fn clear(&mut self) {
        self.entries.clear();
        self.order.clear();
        self.reserved.clear();
        self.tombstones.clear();
        self.retained_bytes = 0;
    }
}

#[cfg(test)]
impl InferenceCache {
    /// Test-only clock seam: rewinds entry insertion times so TTL expiry is
    /// observable without waiting out the real 120-second window.
    pub(crate) fn backdate_entries(&mut self, by: Duration) {
        for entry in self.entries.values_mut() {
            if let Some(earlier) = entry.inserted_at.checked_sub(by) {
                entry.inserted_at = earlier;
            }
        }
    }

    /// Test-only clock seam: rewinds tombstone times so tombstone expiry is
    /// observable without waiting out the real 300-second window.
    pub(crate) fn backdate_tombstones(&mut self, by: Duration) {
        for tombstone in &mut self.tombstones {
            if let Some(earlier) = tombstone.inserted_at.checked_sub(by) {
                tombstone.inserted_at = earlier;
            }
        }
    }
}

pub(crate) fn retained_inference_bytes(result: &NativeInferenceResult) -> Result<usize, ApiError> {
    let feature_bytes = result.features.values().try_fold(0_usize, |total, values| {
        values
            .capacity()
            .checked_mul(std::mem::size_of::<f32>())
            .and_then(|bytes| total.checked_add(bytes))
    });
    let keypoint_bytes = result.keypoints.as_ref().map_or(Some(0), |values| {
        values
            .capacity()
            .checked_mul(std::mem::size_of::<slouch_domain::Keypoint>())
    });
    feature_bytes
        .and_then(|bytes| keypoint_bytes.and_then(|keypoints| bytes.checked_add(keypoints)))
        .and_then(|bytes| bytes.checked_add(std::mem::size_of::<NativeInferenceResult>()))
        .ok_or_else(|| ApiError::InvalidRequest("inference feature bundle size overflows".into()))
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::{InferenceCache, INFERENCE_CACHE_TTL, TOMBSTONE_TTL};
    use slouch_vision::ported::inference_worker::NativeInferenceResult;

    fn result() -> NativeInferenceResult {
        NativeInferenceResult {
            person_found: false,
            bbox: None,
            keypoints: None,
            features: Default::default(),
            classification: None,
        }
    }

    #[test]
    fn entries_expire_after_the_cache_ttl_and_leave_an_expired_tombstone() {
        let mut cache = InferenceCache::with_seed(21);
        let token = cache.insert(3, result()).expect("insert");
        cache.backdate_entries(INFERENCE_CACHE_TTL + Duration::from_secs(1));
        let error = cache.checkout(token, 3).expect_err("expired token");
        assert!(error.to_string().contains("expired"), "{error:?}");
        assert_eq!(
            cache.retained_bytes, 0,
            "expiry must release retained bytes"
        );
    }

    #[test]
    fn expired_tombstones_are_forgotten_after_the_tombstone_ttl() {
        let mut cache = InferenceCache::with_seed(22);
        let token = cache.insert(4, result()).expect("insert");
        cache.backdate_entries(INFERENCE_CACHE_TTL + Duration::from_secs(1));
        let expired = cache.checkout(token, 4).expect_err("expired token");
        assert!(expired.to_string().contains("expired"), "{expired:?}");
        cache.backdate_tombstones(TOMBSTONE_TTL + Duration::from_secs(1));
        let unknown = cache.checkout(token, 4).expect_err("forgotten token");
        assert!(
            unknown.to_string().contains("unknown inference token"),
            "{unknown:?}"
        );
    }
}
