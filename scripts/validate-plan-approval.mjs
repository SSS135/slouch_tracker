import { createHash } from 'node:crypto';
import { existsSync, readFileSync } from 'node:fs';

const sha256 = (path) => createHash('sha256').update(readFileSync(path)).digest('hex');
const path = '.plans/research/rust-refactor-plan-approval.json';
const errors = [];
if (!existsSync(path)) {
  console.error(JSON.stringify({ ok: false, errors: [`missing ${path}`] }, null, 2));
  process.exit(1);
}
const approval = JSON.parse(readFileSync(path, 'utf8'));
const inputs = {
  plan: '.plans/rust-typescript-refactor-plan.md',
  research: '.plans/research/rust-refactor-research-2026-07-14.md',
  portingGuide: 'PORTING.md',
  acceptanceMatrix: '.plans/rust-refactor-acceptance.json',
  readerAgent: '.pi/agents/port-reader.md',
  reviewerAgent: '.pi/agents/port-reviewer.md',
  writerAgent: '.pi/agents/port-writer.md',
  checkerAgent: '.pi/agents/port-checker.md',
  approvalValidator: 'scripts/validate-plan-approval.mjs',
  manifestValidator: 'scripts/validate-port-manifest.mjs',
  acceptanceValidator: 'scripts/validate-acceptance.mjs',
  gateRunner: 'scripts/run-gate.mjs',
  snapshotter: 'scripts/snapshot-tree.mjs',
  globalVerifier: 'scripts/verify-migration.mjs',
};
if (approval.version !== 1 || !approval.inputs || !Array.isArray(approval.reviewers) || !approval.fixer) errors.push('invalid approval shape');
for (const [key, inputPath] of Object.entries(inputs)) {
  if (approval.inputs?.[key]?.path !== inputPath || approval.inputs?.[key]?.sha256 !== sha256(inputPath)) errors.push(`approval input stale: ${key}`);
}
if (approval.reviewers?.length !== 2) errors.push('exactly two plan reviewers required');
const identities = new Set();
for (const [index, review] of (approval.reviewers ?? []).entries()) {
  for (const field of ['identity', 'runId', 'artifact', 'sha256', 'outcome']) if (typeof review[field] !== 'string' || !review[field]) errors.push(`reviewers[${index}].${field} missing`);
  if (identities.has(review.identity)) errors.push('plan reviewer identities must differ');
  identities.add(review.identity);
  if (review.outcome !== 'approved') errors.push(`reviewers[${index}] is not approved`);
  if (!existsSync(review.artifact) || sha256(review.artifact) !== review.sha256) errors.push(`reviewers[${index}] artifact missing/stale`);
  else {
    const verdict = readFileSync(review.artifact, 'utf8').trim().match(/(?:^|\n)(PASS|FAIL)$/)?.[1];
    if (verdict !== 'PASS') errors.push(`reviewers[${index}] artifact terminal verdict is not PASS`);
  }
}
if (typeof approval.fixer.identity !== 'string' || identities.has(approval.fixer.identity) || approval.fixer.outcome !== 'closed' || !Array.isArray(approval.fixer.dispositions) || approval.fixer.dispositions.some((item) => !item.id || !item.disposition)) errors.push('invalid/overlapping plan fixer');
if (approval.blockers?.length) errors.push('plan approval retains blockers');
console.log(JSON.stringify({ ok: errors.length === 0, inputs: Object.keys(inputs).length, reviewers: approval.reviewers?.length ?? 0, errors }, null, 2));
if (errors.length) process.exit(1);
