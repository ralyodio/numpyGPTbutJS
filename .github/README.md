# CI/CD Pipeline Documentation

This directory contains GitHub Actions workflows for automated testing, building, and deployment of the numpyGPT JavaScript implementation.

## Workflows

### 1. Main CI/CD Pipeline (`.github/workflows/ci.yml`)

**Triggers:**
- Push to `main` or `js-implementation` branches
- Pull requests to `main` or `js-implementation` branches

**Jobs:**

#### Test Job
- **Matrix Strategy**: Tests across Node.js versions 18.x, 20.x, and 22.x
- **Steps:**
  - Checkout code
  - Setup Node.js with npm cache
  - Install dependencies
  - Run linting (`npm run lint`)
  - Run tests (`npm test`)
  - Run validation tests (`npm run validate`)
  - Upload test coverage to Codecov (Node.js 20.x only)

#### Build Job
- **Dependencies**: Requires test job to pass
- **Steps:**
  - Build browser version (`npm run build:browser`)
  - Build documentation (`npm run build:docs`)
  - Upload build artifacts (retained for 30 days)

#### Security Job
- **Steps:**
  - Run npm security audit
  - Check for vulnerabilities with audit-ci

#### Performance Job
- **Dependencies**: Requires test job to pass
- **Steps:**
  - Run performance benchmarks (`npm run benchmark`)
  - Upload benchmark results (retained for 7 days)

#### Deploy Documentation Job
- **Triggers**: Only on push to `main` branch
- **Dependencies**: Requires test and build jobs to pass
- **Steps:**
  - Build and deploy documentation to GitHub Pages
  - Custom domain: `numpygpt-js.profullstack.com`

#### Release Job
- **Triggers**: Only on push to `main` branch
- **Dependencies**: Requires test, build, and security jobs to pass
- **Steps:**
  - Semantic release with automated versioning
  - Publish to NPM registry
  - Create GitHub releases with assets

### 2. Pull Request Validation (`.github/workflows/pr.yml`)

**Triggers:**
- Pull requests to `main` or `js-implementation` branches

**Features:**
- Comprehensive validation checks
- Automated PR comments with validation results
- Status indicators for each check type

**Validation Steps:**
- Code linting
- Format checking
- Test execution
- Validation tests
- Browser build verification
- Security audit

## Configuration Files

### Semantic Release (`.releaserc.json`)
- Automated versioning based on conventional commits
- Changelog generation
- NPM publishing
- GitHub release creation with assets
- Support for prerelease versions on `js-implementation` branch

### Package Scripts
Required npm scripts for CI/CD pipeline:
```json
{
  "lint": "eslint src/ *.js",
  "format:check": "prettier --check src/ *.js",
  "test": "jest",
  "validate": "node validate_parity.js",
  "build:browser": "webpack --config webpack.config.js",
  "build:docs": "mkdir -p docs && cp -r examples/ docs/ && cp README.md docs/",
  "benchmark": "node benchmark.js"
}
```

## Environment Variables and Secrets

### Required GitHub Secrets
- `GITHUB_TOKEN`: Automatically provided by GitHub Actions
- `NPM_TOKEN`: Required for NPM publishing (optional)

### Optional Secrets
- `CODECOV_TOKEN`: For enhanced Codecov integration

## Branch Strategy

### Main Branch (`main`)
- Production-ready code
- Triggers full CI/CD pipeline including deployment
- Automated releases and NPM publishing

### Development Branch (`js-implementation`)
- Active development branch
- Triggers CI/CD pipeline with prerelease versions
- Beta releases to NPM with `-beta` suffix

### Pull Requests
- All PRs trigger validation workflow
- Must pass all checks before merge
- Automated status comments on PRs

## Artifacts and Outputs

### Build Artifacts
- Browser bundles (`dist/numpygpt.js`, `dist/numpygpt.min.js`)
- NPM package tarball
- Documentation files
- Retention: 30 days

### Benchmark Results
- Performance metrics and comparisons
- Retention: 7 days

### Test Coverage
- Uploaded to Codecov for tracking
- Coverage reports in PR comments

## Deployment

### GitHub Pages
- Automatic deployment of documentation
- Custom domain support
- Only deploys from `main` branch

### NPM Registry
- Automatic publishing on successful releases
- Semantic versioning
- Support for beta releases

## Monitoring and Notifications

### Status Checks
- All workflows provide status checks for PRs
- Required checks can be configured in branch protection rules

### PR Comments
- Automated validation result comments
- Status indicators for each check type
- Links to detailed workflow logs

## Troubleshooting

### Common Issues

1. **Test Failures**
   - Check Node.js version compatibility
   - Verify all dependencies are installed
   - Review test logs for specific failures

2. **Build Failures**
   - Ensure webpack configuration is correct
   - Check for missing build dependencies
   - Verify all source files are present

3. **Security Audit Failures**
   - Review npm audit output
   - Update vulnerable dependencies
   - Use `npm audit fix` for automatic fixes

4. **Release Failures**
   - Verify semantic release configuration
   - Check commit message format (conventional commits)
   - Ensure all required secrets are configured

### Debug Steps
1. Check workflow logs in GitHub Actions tab
2. Verify package.json scripts are correct
3. Test commands locally before pushing
4. Review branch protection rules and required checks

## Maintenance

### Regular Tasks
- Update Node.js versions in matrix strategy
- Review and update dependencies
- Monitor security audit results
- Update documentation as needed

### Dependency Updates
- Automated dependency updates can be configured with Dependabot
- Security updates should be prioritized
- Test thoroughly after major version updates