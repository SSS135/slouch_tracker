import { createHash } from 'node:crypto';
import { existsSync, readFileSync, readdirSync, statSync } from 'node:fs';
import { dirname, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const manifestPath = resolve(root, process.argv[2] ?? 'src-tauri/port-manifest.json');
const errors = [];
const sha256 = (path) => createHash('sha256').update(readFileSync(path)).digest('hex');
const globPattern = /[*?{}[\]]|(^|[/\\])\.\.([/\\]|$)/;
const requiredInputPaths = [
  '.plans/rust-typescript-refactor-plan.md',
  'PORTING.md',
  'package-lock.json',
  'src-tauri/Cargo.lock',
  'src-tauri/tauri.conf.json',
  'src-tauri/resource-lock.json',
  'src-tauri/model-format-v1.md',
  'src-tauri/schema/live-v1.sql',
  'src-tauri/schema/archive-v1.sql',
];

function walkSource(directory, output = []) {
  if (!existsSync(directory)) return output;
  for (const name of readdirSync(directory).sort()) {
    const absolute = resolve(directory, name);
    const info = statSync(absolute);
    if (info.isDirectory()) walkSource(absolute, output);
    else if (/\.(?:ts|tsx|js|jsx)$/.test(name)) output.push(relative(root, absolute).replaceAll('\\', '/'));
  }
  return output;
}

function walkEvidence(directory, output = []) {
  if (!existsSync(directory)) return output;
  for (const name of readdirSync(directory).sort()) {
    const absolute = resolve(directory, name);
    const info = statSync(absolute);
    if (info.isDirectory()) walkEvidence(absolute, output);
    else output.push(relative(root, absolute).replaceAll('\\', '/'));
  }
  return output;
}

let manifest;
try {
  manifest = JSON.parse(readFileSync(manifestPath, 'utf8'));
} catch (error) {
  console.error(JSON.stringify({ ok: false, errors: [`cannot read manifest: ${error.message}`] }, null, 2));
  process.exit(1);
}

const topFields = ['version', 'workflowRuleVersion', 'migrationState', 'verificationInputs', 'sourceInventory', 'queue'];
for (const field of topFields) if (!(field in manifest)) errors.push(`manifest.${field} is required`);
for (const field of Object.keys(manifest)) if (!topFields.includes(field)) errors.push(`manifest.${field} is unknown`);
if (manifest.version !== 2) errors.push('manifest.version must be 2');
if (!Number.isInteger(manifest.workflowRuleVersion) || manifest.workflowRuleVersion < 1) errors.push('manifest.workflowRuleVersion must be a positive integer');
if (!['porting', 'clean-cut'].includes(manifest.migrationState)) errors.push('manifest.migrationState must be porting or clean-cut');

if (!manifest.verificationInputs || typeof manifest.verificationInputs !== 'object' || Array.isArray(manifest.verificationInputs)) {
  errors.push('manifest.verificationInputs must be an object');
} else {
  for (const path of requiredInputPaths) {
    const expected = manifest.verificationInputs[path];
    const absolute = resolve(root, path);
    if (typeof expected !== 'string') errors.push(`verificationInputs missing ${path}`);
    else if (!existsSync(absolute)) errors.push(`verification input missing on disk: ${path}`);
    else if (sha256(absolute) !== expected) errors.push(`verification input hash stale: ${path}`);
  }
  for (const path of Object.keys(manifest.verificationInputs)) if (!requiredInputPaths.includes(path)) errors.push(`verificationInputs unknown path: ${path}`);
}

if (!Array.isArray(manifest.queue)) errors.push('manifest.queue must be an array');
const requiredEntryFields = [
  'id', 'sourcePaths', 'targetPaths', 'category', 'dependsOn', 'fixtureIds',
  'semanticRisks', 'workflowRuleVersion', 'phase', 'status', 'implementer',
  'reviews', 'fixer', 'compiler', 'parity', 'blockers', 'residualRisks',
  'invalidationReason', 'history',
];
const allowedStatus = new Set(['queued', 'drafted', 'implemented', 'review-pending', 'fix-pending', 'verified', 'blocked', 'invalidated']);
const entries = Array.isArray(manifest.queue) ? manifest.queue : [];
const ids = new Set();
const byId = new Map(entries.map((entry) => [entry.id, entry]));
const referencedReviews = new Set();
const referencedFixtures = new Set();
const targetOwners = new Map();
const referencedEvidence = new Set();
const evidenceCache = new Map();
const transitionMap = {
  queued: new Set(['drafted', 'blocked', 'invalidated']),
  drafted: new Set(['implemented', 'blocked', 'invalidated']),
  implemented: new Set(['review-pending', 'blocked', 'invalidated']),
  'review-pending': new Set(['fix-pending', 'verified', 'blocked', 'invalidated']),
  'fix-pending': new Set(['review-pending', 'verified', 'blocked', 'invalidated']),
  blocked: new Set(['queued', 'drafted', 'implemented', 'review-pending', 'fix-pending', 'invalidated']),
  invalidated: new Set(['queued']),
  verified: new Set(['invalidated']),
};
function ownEvidence(path) {
  if (path) referencedEvidence.add(path);
}
function loadEvidence(path) {
  if (evidenceCache.has(path)) return evidenceCache.get(path);
  const absolute = resolve(root, path);
  if (!existsSync(absolute)) {
    errors.push(`evidence artifact missing: ${path}`);
    evidenceCache.set(path, null);
    return null;
  }
  try {
    const evidence = JSON.parse(readFileSync(absolute, 'utf8'));
    evidenceCache.set(path, evidence);
    return evidence;
  } catch (error) {
    errors.push(`evidence artifact is not valid JSON: ${path}: ${error.message}`);
    evidenceCache.set(path, null);
    return null;
  }
}
function checkFields(value, allowed, prefix) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return;
  for (const field of Object.keys(value)) if (!allowed.includes(field)) errors.push(`${prefix}.${field} unknown`);
}
function evidenceEntry(path, entryId) {
  const evidence = loadEvidence(path);
  return Array.isArray(evidence?.entries) ? evidence.entries.find((entry) => entry.entryId === entryId) : null;
}
function validateEvidence(path, kind, entryId, identity, expected = {}) {
  const evidence = loadEvidence(path);
  if (!evidence) return;
  const fields = ['version', 'kind', 'workflowRuleVersion', 'runId', 'agentLabel', 'agentType', 'identity', 'inputFingerprint', 'entries'];
  for (const field of fields) if (!(field in evidence)) errors.push(`${path} evidence.${field} missing`);
  for (const field of Object.keys(evidence)) if (!fields.includes(field)) errors.push(`${path} evidence.${field} unknown`);
  if (evidence.version !== 1 || evidence.workflowRuleVersion !== manifest.workflowRuleVersion || evidence.kind !== kind) errors.push(`${path} evidence header mismatch`);
  if (typeof evidence.runId !== 'string' || !evidence.runId || typeof evidence.agentLabel !== 'string' || !evidence.agentLabel || typeof evidence.agentType !== 'string' || !evidence.agentType || typeof evidence.inputFingerprint !== 'string' || !evidence.inputFingerprint) errors.push(`${path} evidence workflow identity/fingerprint missing`);
  if (identity && evidence.identity !== identity) errors.push(`${path} evidence identity mismatch for ${entryId}`);
  if (expected.inputFingerprint && evidence.inputFingerprint !== expected.inputFingerprint) errors.push(`${path} evidence input fingerprint mismatch for ${entryId}`);
  const item = Array.isArray(evidence.entries) ? evidence.entries.find((entry) => entry.entryId === entryId) : null;
  if (!item) {
    errors.push(`${path} has no evidence entry for ${entryId}`);
    return;
  }
  if (kind === 'review') {
    checkFields(item, ['entryId', 'outcome', 'sourceHashes', 'targetHashes', 'findings'], `${path}:${entryId}`);
    if (!['approved', 'changes-required'].includes(item.outcome) || !item.sourceHashes || !item.targetHashes || !Array.isArray(item.findings)) errors.push(`${path} invalid review entry for ${entryId}`);
    const manifestEntry = byId.get(entryId);
    for (const source of manifestEntry?.sourcePaths ?? []) if (item.sourceHashes?.[source.path] !== source.sha256) errors.push(`${path} review source hash mismatch for ${source.path}`);
    for (const target of manifestEntry?.targetPaths ?? []) if (existsSync(resolve(root, target)) && item.targetHashes?.[target] !== sha256(resolve(root, target))) errors.push(`${path} review target hash mismatch for ${target}`);
    const findingIds = new Set();
    for (const finding of item.findings ?? []) {
      checkFields(finding, ['id', 'severity', 'source', 'target', 'divergence', 'fix', 'verification'], `${path}:${entryId}:finding`);
      for (const field of ['id', 'severity', 'source', 'target', 'divergence', 'fix', 'verification']) if (typeof finding?.[field] !== 'string') errors.push(`${path} invalid review finding for ${entryId}`);
      if (findingIds.has(finding.id)) errors.push(`${path} duplicate finding ID ${finding.id} for ${entryId}`);
      findingIds.add(finding.id);
    }
    if (expected.outcome && item.outcome !== expected.outcome) errors.push(`${path} review outcome mismatch for ${entryId}`);
  } else if (kind === 'implementer') {
    checkFields(item, ['entryId', 'sourceHashes', 'targetHashes', 'changedPaths'], `${path}:${entryId}`);
    if (!item.sourceHashes || !item.targetHashes || !Array.isArray(item.changedPaths)) errors.push(`${path} invalid implementer entry for ${entryId}`);
    const manifestEntry = byId.get(entryId);
    for (const source of manifestEntry?.sourcePaths ?? []) if (item.sourceHashes?.[source.path] !== source.sha256) errors.push(`${path} implementer source hash mismatch for ${source.path}`);
    for (const target of manifestEntry?.targetPaths ?? []) if (existsSync(resolve(root, target)) && item.targetHashes?.[target] !== sha256(resolve(root, target))) errors.push(`${path} implementer target hash mismatch for ${target}`);
  } else if (kind === 'fixer') {
    checkFields(item, ['entryId', 'acceptedFindings', 'rejectedFindings', 'targetHashes'], `${path}:${entryId}`);
    if (!Array.isArray(item.acceptedFindings) || !Array.isArray(item.rejectedFindings) || !item.targetHashes || item.acceptedFindings.some((id) => typeof id !== 'string') || item.rejectedFindings.some((finding) => typeof finding?.id !== 'string' || typeof finding?.reason !== 'string')) errors.push(`${path} invalid fixer entry for ${entryId}`);
    for (const target of byId.get(entryId)?.targetPaths ?? []) if (existsSync(resolve(root, target)) && item.targetHashes?.[target] !== sha256(resolve(root, target))) errors.push(`${path} fixer target hash mismatch for ${target}`);
  } else if (kind === 'compiler' || kind === 'parity') {
    checkFields(item, ['entryId', 'command', 'outcome', 'inputFingerprint', 'outputFingerprint'], `${path}:${entryId}`);
    if (item.command !== expected.command || item.outcome !== expected.outcome || typeof item.inputFingerprint !== 'string' || typeof item.outputFingerprint !== 'string') errors.push(`${path} invalid ${kind} entry for ${entryId}`);
  } else if (kind === 'transition') {
    checkFields(item, ['entryId', 'from', 'to'], `${path}:${entryId}`);
    if (item.from !== expected.from || item.to !== expected.to) errors.push(`${path} invalid transition entry for ${entryId}`);
  }
}

for (const [index, entry] of entries.entries()) {
  const prefix = `queue[${index}]`;
  if (!entry || typeof entry !== 'object' || Array.isArray(entry)) {
    errors.push(`${prefix} must be an object`);
    continue;
  }
  for (const key of requiredEntryFields) if (!(key in entry)) errors.push(`${prefix}.${key} is required`);
  for (const key of Object.keys(entry)) if (!requiredEntryFields.includes(key)) errors.push(`${prefix}.${key} is unknown`);
  if (typeof entry.id !== 'string' || !/^[a-z0-9][a-z0-9._-]*$/.test(entry.id)) errors.push(`${prefix}.id is invalid`);
  else if (ids.has(entry.id)) errors.push(`${prefix}.id duplicates ${entry.id}`);
  else ids.add(entry.id);
  if (!allowedStatus.has(entry.status)) errors.push(`${prefix}.status is invalid`);
  if (entry.workflowRuleVersion !== manifest.workflowRuleVersion) errors.push(`${prefix}.workflowRuleVersion differs from manifest`);

  if (!Array.isArray(entry.sourcePaths) || entry.sourcePaths.length === 0) errors.push(`${prefix}.sourcePaths must be nonempty`);
  else for (const [sourceIndex, source] of entry.sourcePaths.entries()) {
    const sourcePrefix = `${prefix}.sourcePaths[${sourceIndex}]`;
    checkFields(source, ['path', 'sha256'], sourcePrefix);
    if (!source || typeof source.path !== 'string' || typeof source.sha256 !== 'string') {
      errors.push(`${sourcePrefix} must contain path and sha256`);
      continue;
    }
    if (globPattern.test(source.path) || /remaining/i.test(source.path)) errors.push(`${sourcePrefix}.path must be concrete`);
    const absolute = resolve(root, source.path);
    if (!existsSync(absolute) && manifest.migrationState !== 'clean-cut') errors.push(`${sourcePrefix}.path does not exist: ${source.path}`);
    else if (existsSync(absolute) && sha256(absolute) !== source.sha256) errors.push(`${sourcePrefix}.sha256 is stale: ${source.path}`);
  }

  if (!Array.isArray(entry.targetPaths) || entry.targetPaths.length === 0) errors.push(`${prefix}.targetPaths must be nonempty`);
  else for (const target of entry.targetPaths) {
    if (typeof target !== 'string' || globPattern.test(target) || /remaining/i.test(target)) errors.push(`${prefix}.targetPaths must be concrete`);
    else {
      if (targetOwners.has(target)) errors.push(`${prefix}.targetPaths reuses ${target} owned by ${targetOwners.get(target)}`);
      else targetOwners.set(target, entry.id);
      if (entry.status === 'verified' && !existsSync(resolve(root, target))) errors.push(`${prefix}.target missing for verified entry: ${target}`);
    }
  }

  for (const key of ['dependsOn', 'fixtureIds', 'semanticRisks', 'blockers', 'residualRisks']) {
    if (!Array.isArray(entry[key])) errors.push(`${prefix}.${key} must be an array`);
  }
  for (const fixture of entry.fixtureIds ?? []) {
    if (typeof fixture !== 'string' || globPattern.test(fixture)) errors.push(`${prefix}.fixtureIds contains invalid path`);
    else {
      referencedFixtures.add(fixture);
      if (!['queued', 'drafted'].includes(entry.status) && !existsSync(resolve(root, fixture))) errors.push(`${prefix} fixture missing: ${fixture}`);
    }
  }

  const implementer = entry.implementer;
  checkFields(implementer, ['identity', 'artifact'], `${prefix}.implementer`);
  if (implementer !== null && (typeof implementer?.identity !== 'string' || typeof implementer?.artifact !== 'string')) errors.push(`${prefix}.implementer must be null or {identity,artifact}`);
  if (implementer?.artifact) {
    ownEvidence(implementer.artifact);
    validateEvidence(implementer.artifact, 'implementer', entry.id, implementer.identity);
  }

  if (!entry.reviews || typeof entry.reviews !== 'object' || !('one' in entry.reviews) || !('two' in entry.reviews)) errors.push(`${prefix}.reviews must contain one and two`);
  else for (const side of ['one', 'two']) {
    const review = entry.reviews[side];
    checkFields(review, ['reviewer', 'artifact', 'inputFingerprint', 'outcome'], `${prefix}.reviews.${side}`);
    if (review !== null && (typeof review?.reviewer !== 'string' || typeof review?.artifact !== 'string' || typeof review?.inputFingerprint !== 'string' || !['approved', 'changes-required'].includes(review?.outcome))) errors.push(`${prefix}.reviews.${side} has invalid nested schema`);
    if (review?.artifact) {
      referencedReviews.add(review.artifact);
      ownEvidence(review.artifact);
      validateEvidence(review.artifact, 'review', entry.id, review.reviewer, { outcome: review.outcome, inputFingerprint: review.inputFingerprint });
    }
  }

  const fixer = entry.fixer;
  checkFields(fixer, ['identity', 'artifact', 'outcome'], `${prefix}.fixer`);
  if (fixer !== null && (typeof fixer?.identity !== 'string' || typeof fixer?.artifact !== 'string' || !['open', 'closed'].includes(fixer?.outcome))) errors.push(`${prefix}.fixer has invalid nested schema`);
  if (fixer?.artifact) {
    ownEvidence(fixer.artifact);
    validateEvidence(fixer.artifact, 'fixer', entry.id, fixer.identity);
  }
  for (const key of ['compiler', 'parity']) {
    const evidence = entry[key];
    checkFields(evidence, ['command', 'outcome', 'evidence'], `${prefix}.${key}`);
    if (!evidence || typeof evidence.command !== 'string' || !['pending', 'passed', 'failed', 'blocked'].includes(evidence.outcome) || (evidence.evidence !== null && typeof evidence.evidence !== 'string')) errors.push(`${prefix}.${key} has invalid nested schema`);
    if (evidence?.evidence) {
      ownEvidence(evidence.evidence);
      validateEvidence(evidence.evidence, key, entry.id, null, { command: evidence.command, outcome: evidence.outcome });
    }
  }

  if (!Array.isArray(entry.history) || entry.history.length === 0) errors.push(`${prefix}.history must be nonempty`);
  else {
    for (const [historyIndex, transition] of entry.history.entries()) {
      checkFields(transition, ['from', 'to', 'evidence'], `${prefix}.history[${historyIndex}]`);
      if (!transition || !('from' in transition) || !('to' in transition) || typeof transition.evidence !== 'string') errors.push(`${prefix}.history[${historyIndex}] has invalid schema`);
      if (historyIndex === 0 && transition.from !== null) errors.push(`${prefix}.history must start from null`);
      if (historyIndex === 0 && transition.to !== 'queued') errors.push(`${prefix}.history must start at queued`);
      if (historyIndex > 0) {
        const previous = entry.history[historyIndex - 1];
        if (transition.from !== previous.to) errors.push(`${prefix}.history[${historyIndex}] does not continue prior state`);
        if (!transitionMap[transition.from]?.has(transition.to)) errors.push(`${prefix}.history[${historyIndex}] invalid transition ${transition.from} -> ${transition.to}`);
      }
      ownEvidence(transition.evidence);
      validateEvidence(transition.evidence, 'transition', entry.id, null, { from: transition.from, to: transition.to });
    }
    if (entry.history.at(-1)?.to !== entry.status) errors.push(`${prefix}.history final state differs from status`);
  }

  const identities = [implementer?.identity, entry.reviews?.one?.reviewer, entry.reviews?.two?.reviewer, fixer?.identity].filter(Boolean);
  if (new Set(identities).size !== identities.length) errors.push(`${prefix} reuses implementer/reviewer/fixer identity`);
  if (fixer?.artifact && entry.reviews?.one?.artifact && entry.reviews?.two?.artifact) {
    const findingIds = new Set([
      ...(evidenceEntry(entry.reviews.one.artifact, entry.id)?.findings ?? []).map((finding) => finding.id),
      ...(evidenceEntry(entry.reviews.two.artifact, entry.id)?.findings ?? []).map((finding) => finding.id),
    ]);
    const fixerEntry = evidenceEntry(fixer.artifact, entry.id);
    const dispositions = [
      ...(fixerEntry?.acceptedFindings ?? []),
      ...(fixerEntry?.rejectedFindings ?? []).map((finding) => finding.id),
    ];
    if (new Set(dispositions).size !== dispositions.length) errors.push(`${prefix} fixer duplicates finding disposition`);
    for (const id of findingIds) if (!dispositions.includes(id)) errors.push(`${prefix} fixer does not dispose finding ${id}`);
    for (const id of dispositions) if (!findingIds.has(id)) errors.push(`${prefix} fixer disposes unknown finding ${id}`);
  }

  if (entry.status === 'verified') {
    if (!implementer?.artifact || !existsSync(resolve(root, implementer.artifact))) errors.push(`${prefix} verified without implementer evidence`);
    for (const side of ['one', 'two']) {
      const review = entry.reviews?.[side];
      if (!review?.reviewer || !review?.artifact || !['approved', 'changes-required'].includes(review?.outcome)) errors.push(`${prefix} verified without completed review ${side}`);
      else if (!existsSync(resolve(root, review.artifact))) errors.push(`${prefix} review artifact missing: ${review.artifact}`);
    }
    if (!fixer?.artifact || fixer?.outcome !== 'closed' || !existsSync(resolve(root, fixer.artifact))) errors.push(`${prefix} verified without closed fixer evidence`);
    for (const key of ['compiler', 'parity']) {
      if (entry[key]?.outcome !== 'passed' || !entry[key]?.command || !entry[key]?.evidence || !existsSync(resolve(root, entry[key].evidence))) errors.push(`${prefix} verified without passed ${key} command/evidence`);
    }
    if (entry.blockers?.length) errors.push(`${prefix} verified with blockers`);
    if (entry.residualRisks?.length) errors.push(`${prefix} verified with residual risks`);
    if (entry.invalidationReason) errors.push(`${prefix} verified with invalidationReason`);
  }
}

for (const [index, entry] of entries.entries()) {
  for (const dependency of entry.dependsOn ?? []) if (!ids.has(dependency)) errors.push(`queue[${index}].dependsOn references missing ${dependency}`);
}

const visiting = new Set();
const visited = new Set();
function visit(id, trail = []) {
  if (visiting.has(id)) {
    errors.push(`dependency cycle: ${[...trail, id].join(' -> ')}`);
    return;
  }
  if (visited.has(id) || !byId.has(id)) return;
  visiting.add(id);
  for (const dependency of byId.get(id).dependsOn ?? []) visit(dependency, [...trail, id]);
  visiting.delete(id);
  visited.add(id);
}
for (const id of ids) visit(id);

if (!Array.isArray(manifest.sourceInventory)) errors.push('manifest.sourceInventory must be an array');
const inventory = Array.isArray(manifest.sourceInventory) ? manifest.sourceInventory : [];
const inventoryByPath = new Map();
const allowedDisposition = new Set(['port', 'retain-ui', 'remove-web']);
for (const [index, item] of inventory.entries()) {
  const prefix = `sourceInventory[${index}]`;
  const fields = ['path', 'sha256', 'disposition', 'entryId', 'reason'];
  if (!item || typeof item !== 'object') {
    errors.push(`${prefix} must be an object`);
    continue;
  }
  for (const field of fields) if (!(field in item)) errors.push(`${prefix}.${field} is required`);
  for (const field of Object.keys(item)) if (!fields.includes(field)) errors.push(`${prefix}.${field} is unknown`);
  if (inventoryByPath.has(item.path)) errors.push(`${prefix}.path duplicates ${item.path}`);
  else inventoryByPath.set(item.path, item);
  if (!allowedDisposition.has(item.disposition)) errors.push(`${prefix}.disposition is invalid`);
  const absolute = resolve(root, item.path ?? '');
  const present = existsSync(absolute);
  if (manifest.migrationState === 'porting') {
    if (!present) errors.push(`${prefix}.path does not exist: ${item.path}`);
    else if (sha256(absolute) !== item.sha256) errors.push(`${prefix}.sha256 is stale: ${item.path}`);
  } else if (item.disposition === 'retain-ui') {
    if (!present) errors.push(`${prefix}.retain-ui path was deleted: ${item.path}`);
    else if (sha256(absolute) !== item.sha256) errors.push(`${prefix}.retain-ui hash is stale: ${item.path}`);
  } else if (present) errors.push(`${prefix}.${item.disposition} source remains after clean cut: ${item.path}`);
  if (item.disposition === 'port') {
    if (!ids.has(item.entryId)) errors.push(`${prefix}.entryId missing from queue: ${item.entryId}`);
    else if (!(byId.get(item.entryId).sourcePaths ?? []).some((source) => source.path === item.path)) errors.push(`${prefix} is not listed by entry ${item.entryId}`);
  } else {
    if (item.entryId !== null) errors.push(`${prefix}.entryId must be null for ${item.disposition}`);
    if (typeof item.reason !== 'string' || !item.reason.trim()) errors.push(`${prefix}.reason is required for ${item.disposition}`);
  }
}

const sourceFiles = walkSource(resolve(root, 'src'));
for (const path of sourceFiles) {
  const item = inventoryByPath.get(path);
  if (item) {
    if (manifest.migrationState === 'clean-cut' && item.disposition !== 'retain-ui') errors.push(`migrated/removed source remains after clean cut: ${path}`);
  } else {
    const owner = targetOwners.get(path);
    if (!owner || byId.get(owner)?.status !== 'verified') errors.push(`current source is neither inventoried nor a verified generated target: ${path}`);
  }
}
if (manifest.migrationState === 'porting') for (const path of inventoryByPath.keys()) if (!sourceFiles.includes(path)) errors.push(`sourceInventory path outside required src JS/TS inventory: ${path}`);
if (manifest.migrationState === 'clean-cut') for (const item of inventory) {
  if (item.disposition === 'port' && byId.get(item.entryId)?.status !== 'verified') errors.push(`deleted port source is not verified: ${item.path}`);
}
for (const entry of entries) for (const source of entry.sourcePaths ?? []) {
  const item = inventoryByPath.get(source.path);
  if (!item || item.disposition !== 'port' || item.entryId !== entry.id) errors.push(`queue source lacks matching port inventory: ${entry.id} -> ${source.path}`);
}

for (const evidence of walkEvidence(resolve(root, 'src-tauri/reviews'))) if (!evidence.startsWith('src-tauri/reviews/acceptance/') && !referencedEvidence.has(evidence)) errors.push(`orphan review/evidence artifact: ${evidence}`);
for (const fixture of walkEvidence(resolve(root, 'src-tauri/fixtures'))) {
  const id = fixture.replace(/^src-tauri\/fixtures\//, '');
  if (!referencedFixtures.has(id) && !referencedFixtures.has(fixture)) errors.push(`orphan fixture artifact: ${fixture}`);
}

const result = { ok: errors.length === 0, manifest: manifestPath, entries: entries.length, sources: inventory.length, errors };
console.log(JSON.stringify(result, null, 2));
if (!result.ok) process.exit(1);
