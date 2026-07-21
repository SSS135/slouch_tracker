# CMD Guidelines

This file tracks successful test commands and patterns for running tests in this project.

## Test Commands

### IMPORTANT: Vitest Syntax
This project uses Vitest (not Jest). Use file path patterns without `--testPathPatterns` flag:
```bash
npm test -- useFrameSampler.test --no-coverage
npm test -- src/services/ml --no-coverage
```

### Run all tests
```bash
npm test -- --no-coverage
```

### Run specific test file
```bash
npm test -- logisticRegressionClassifier.test --no-coverage
npm test -- svmClassifier.test --no-coverage
```

### Run tests in a directory
```bash
npm test -- src/components/dataset --no-coverage
npm test -- src/services/ml --no-coverage
```

## Notes

- This project uses Vitest (not Jest) - use simple file path patterns
- Use `--no-coverage` for faster test runs during development
- Test files are located in `__tests__` directories or with `.test.ts` suffix

## Dependencies

- `@testing-library/jest-dom` is required in vitest.setup.ts
- Install with: `npm install --save-dev @testing-library/jest-dom`

## Vitest Configuration

- Uses native ES modules
- Web Worker mocks may cause some async test issues (known limitation)

## Git Worktrees

Use git worktrees for isolated feature development to avoid switching branches in main directory. Only if requested by user.

### Create worktree for new feature
```bash
git worktree add ../slouch_tracker_feature -b feature/name master
```

### List worktrees
```bash
git worktree list
```

### Remove worktree after merging
```bash
git worktree remove ../slouch_tracker_feature
git worktree prune
```

### Workflow
1. Create worktree with new branch from master
2. Work in worktree directory
3. Commit changes, rebase on master if needed
4. Merge to master from main repo
5. Remove worktree

Note: If `git worktree remove` fails with "directory in use", close all processes accessing the worktree directory (editors, terminals, file explorers) and retry.