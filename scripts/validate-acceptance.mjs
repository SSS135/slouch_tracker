import { spawnSync } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';

const final = process.argv.includes('--final');
const matrix = JSON.parse(readFileSync('.plans/rust-refactor-acceptance.json', 'utf8'));
const requiredIds = [
  'manifest',
  'rust-static',
  'rust-test',
  'compatibility-oracles',
  'app-integration',
  'svelte-pre-cutover',
  'frontend',
  'bindings',
  'browser-e2e',
  'native-e2e',
  'package',
  'security-data',
  'cleanup',
  'end-to-end',
];
const errors = [];
const topFields = ['version', 'target', 'sourceFingerprint', 'rows'];
for (const field of topFields) if (!(field in matrix)) errors.push(`matrix.${field} missing`);
for (const field of Object.keys(matrix)) if (!topFields.includes(field)) errors.push(`matrix.${field} unknown`);
if (matrix.version !== 2 || !Array.isArray(matrix.rows)) errors.push('invalid acceptance matrix shape');
if (matrix.sourceFingerprint !== null && !/^[a-f0-9]{64}$/.test(matrix.sourceFingerprint)) errors.push('matrix.sourceFingerprint must be null or a SHA-256 digest');

const snapshotRun = spawnSync('node scripts/snapshot-tree.mjs', { shell: true, encoding: 'utf8' });
let currentFingerprint = '';
try {
  currentFingerprint = JSON.parse(snapshotRun.stdout || '{}').digest ?? '';
} catch {
  errors.push('current source fingerprint is unreadable');
}
if (snapshotRun.status !== 0 || !/^[a-f0-9]{64}$/.test(currentFingerprint)) errors.push('current source fingerprint failed');
if (final && matrix.sourceFingerprint !== currentFingerprint) errors.push('acceptance matrix source fingerprint drift');

const ids = new Set();
for (const [index, row] of (matrix.rows ?? []).entries()) {
  for (const field of ['id', 'platform', 'setup', 'command', 'expected', 'evidence', 'manifestEntryIds', 'blocking', 'status']) if (!(field in row)) errors.push(`rows[${index}].${field} missing`);
  if (ids.has(row.id)) errors.push(`duplicate row ${row.id}`);
  ids.add(row.id);
  if (row.blocking !== true) errors.push(`${row.id} must be blocking for Windows migration`);
  if (!['pending', 'passed', 'failed', 'blocked'].includes(row.status)) errors.push(`${row.id} invalid status`);
  if (!Array.isArray(row.manifestEntryIds)) errors.push(`${row.id} manifestEntryIds must be an array`);
  if (final && row.manifestEntryIds.length === 0) errors.push(`${row.id} manifestEntryIds is empty`);
  if (final && row.status !== 'passed') errors.push(`${row.id} is not passed`);
  if (final && (!row.evidence || !existsSync(row.evidence))) errors.push(`${row.id} evidence missing: ${row.evidence}`);
  else if (final) {
    try {
      const evidence = JSON.parse(readFileSync(row.evidence, 'utf8'));
      const fields = ['version', 'rowId', 'command', 'manifestEntryIds', 'ok', 'runId', 'agentLabel', 'inputFingerprint', 'outputFingerprint', 'results'];
      for (const field of fields) if (!(field in evidence)) errors.push(`${row.id} evidence.${field} missing`);
      for (const field of Object.keys(evidence)) if (!fields.includes(field)) errors.push(`${row.id} evidence.${field} unknown`);
      if (evidence.version !== 1 || evidence.rowId !== row.id || evidence.command !== row.command || evidence.ok !== true || JSON.stringify(evidence.manifestEntryIds) !== JSON.stringify(row.manifestEntryIds)) errors.push(`${row.id} evidence header mismatch`);
      if (typeof evidence.runId !== 'string' || !evidence.runId || typeof evidence.agentLabel !== 'string' || !evidence.agentLabel) errors.push(`${row.id} evidence workflow identity missing`);
      if (evidence.inputFingerprint !== matrix.sourceFingerprint || evidence.outputFingerprint !== matrix.sourceFingerprint) errors.push(`${row.id} evidence source fingerprint drift`);
      if (!Array.isArray(evidence.results) || evidence.results.length === 0 || evidence.results.some((result) => result.ok !== true || typeof result.command !== 'string' || typeof result.status !== 'number') || !evidence.results.some((result) => result.command === row.command && result.ok === true)) errors.push(`${row.id} evidence has failed/empty/unmatched results`);
    } catch (error) {
      errors.push(`${row.id} evidence invalid JSON: ${error.message}`);
    }
  }
}
for (const id of requiredIds) if (!ids.has(id)) errors.push(`required row missing: ${id}`);
console.log(JSON.stringify({ ok: errors.length === 0, final, sourceFingerprint: matrix.sourceFingerprint, currentFingerprint, rows: matrix.rows?.length ?? 0, errors }, null, 2));
if (errors.length) process.exit(1);
