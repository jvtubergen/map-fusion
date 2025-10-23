# Release Quick Start Guide

Quick reference for creating and publishing map_fusion data releases.

## Prerequisites

- Python 3.11+ with hatch installed
- Git repository with push access
- GitHub repository with Actions enabled

## Creating a Release (Step by Step)

### 1. Test the Build Locally

```bash
# Quick validation (uses test mode)
bash build_release_data.sh --test --clean

# Full build (downloads real GPS data)
bash build_release_data.sh --clean

# Check output
ls -lh build/release/
```

### 2. Update Version

Edit `VERSION` file to set the release version:

```bash
# For next alpha release
echo "0.1.0-alpha.2" > VERSION
```

### 3. Update Changelog

Edit `CHANGELOG.md` to document changes:

```markdown
## [0.1.0-alpha.2] - 2025-10-24

### Added
- Improved GPS data processing performance

### Fixed
- Fixed coordinate transformation bug for Chicago region
```

### 4. Commit Changes

```bash
git add VERSION CHANGELOG.md
git commit -m "Release v0.1.0-alpha.2"
git push origin main
```

### 5. Create and Push Tag

```bash
# Create annotated tag
git tag -a v0.1.0-alpha.2 -m "Release v0.1.0-alpha.2

- Improved GPS data processing performance
- Fixed coordinate transformation bug"

# Push tag to trigger release workflow
git push origin v0.1.0-alpha.2
```

### 6. Monitor GitHub Actions

1. Go to: `https://github.com/USERNAME/map-fusion/actions`
2. Watch the "Build and Release" workflow
3. Check for any errors

### 7. Verify Release

1. Go to: `https://github.com/USERNAME/map-fusion/releases`
2. Find your release (marked as pre-release for alpha/beta)
3. Download and verify the tarball:

```bash
wget https://github.com/USERNAME/map-fusion/releases/download/v0.1.0-alpha.2/release-v0.1.0-alpha.2.tar.gz
wget https://github.com/USERNAME/map-fusion/releases/download/v0.1.0-alpha.2/release-v0.1.0-alpha.2.tar.gz.sha256

# Verify checksum
sha256sum -c release-v0.1.0-alpha.2.tar.gz.sha256

# Extract and inspect
tar -xzf release-v0.1.0-alpha.2.tar.gz
cd release-v0.1.0-alpha.2
cat README.md
```

### 8. Update Version to Development

```bash
# Bump to next development version
echo "0.1.0-alpha.3-dev" > VERSION
git add VERSION
git commit -m "Bump version to 0.1.0-alpha.3-dev"
git push origin main
```

## Common Commands

### Build Commands

```bash
# Full clean build
bash build_release_data.sh --clean

# Test mode (fast, for validation)
bash build_release_data.sh --test

# Build specific stage only
bash build_release_data.sh --stage input

# Override version
bash build_release_data.sh --version 0.1.0-alpha.99

# Custom output directory
bash build_release_data.sh --output /tmp/release-build
```

### Git Commands

```bash
# View tags
git tag -l "v*"

# View tag details
git show v0.1.0-alpha.2

# Delete local tag (if needed)
git tag -d v0.1.0-alpha.2

# Delete remote tag (use carefully!)
git push origin --delete v0.1.0-alpha.2
```

### GitHub CLI Commands

```bash
# List releases
gh release list

# View specific release
gh release view v0.1.0-alpha.2

# Download release assets
gh release download v0.1.0-alpha.2

# Delete release (use carefully!)
gh release delete v0.1.0-alpha.2
```

## Version Numbering Guide

### Alpha Releases (Current)

```
v0.1.0-alpha.1   # First alpha - GPS only
v0.1.0-alpha.2   # Bug fixes, improvements to GPS
v0.2.0-alpha.1   # GPS + satellite input added
v0.3.0-alpha.1   # GPS + satellite + unimodal
```

**When to increment**:
- **Patch** (0.1.X): Bug fixes, minor improvements
- **Minor** (0.X.0): New pipeline stage or major feature
- **Major** (X.0.0): Breaking changes (unlikely before v1.0)
- **Alpha number**: Sequential within same minor version

### Future Versions

```
v0.5.0-beta.1    # First beta (all stages working)
v0.5.0-beta.2    # Beta refinements
v0.5.0-rc.1      # Release candidate
v1.0.0           # First stable release
```

## Troubleshooting

### Build Fails Locally

```bash
# Check dependencies
hatch env prune
hatch shell
python -m map_fusion.pipeline.input.gps --help

# Check disk space
df -h

# Check permissions
ls -la build/
```

### GitHub Actions Fails

1. Check workflow logs in Actions tab
2. Common issues:
   - VERSION file doesn't match tag
   - Missing dependencies in CI
   - Insufficient permissions
   - Network timeout downloading GPS data

### Version Mismatch Error

```
ERROR: VERSION file (0.1.0-alpha.1) does not match git tag (0.1.0-alpha.2)
```

**Fix**: Update VERSION file to match the tag you're creating:
```bash
echo "0.1.0-alpha.2" > VERSION
git add VERSION
git commit --amend -m "Release v0.1.0-alpha.2"
git push origin main --force-with-lease
```

### Tag Already Exists

```bash
# Delete local tag
git tag -d v0.1.0-alpha.2

# Delete remote tag
git push origin --delete v0.1.0-alpha.2

# Recreate tag
git tag -a v0.1.0-alpha.2 -m "Release message"
git push origin v0.1.0-alpha.2
```

## Testing Releases

### Manual Testing Workflow

Before creating official release:

```bash
# 1. Build locally in test mode
bash build_release_data.sh --test --clean

# 2. Inspect artifact
cd build/release
tar -tzf release-*.tar.gz | head -20
tar -xzf release-*.tar.gz

# 3. Check structure
ls -la release-v*/
cat release-v*/README.md
cat release-v*/metadata.json

# 4. Verify GPS data
ls -la release-v*/input/gps/berlin/
ls -la release-v*/input/gps/chicago/
```

### Automated Testing (Future)

The GitHub Actions workflow includes a validation job. Future enhancements:

- Extract and verify tarball structure
- Validate GPS data format
- Check file sizes and counts
- Run smoke tests on data
- Compare against previous release

## Release Checklist

Use this checklist for each release:

- [ ] All code changes committed and pushed
- [ ] Tests pass: `python -m pytest tests/`
- [ ] Type checking passes: `hatch run types:check`
- [ ] Build works locally: `bash build_release_data.sh --test`
- [ ] VERSION file updated (remove `-dev` suffix)
- [ ] CHANGELOG.md updated with changes
- [ ] Git tag created with correct version
- [ ] Tag pushed to GitHub
- [ ] GitHub Actions workflow succeeded
- [ ] Release artifact uploaded to GitHub
- [ ] Tarball downloaded and verified locally
- [ ] VERSION file bumped to next `-dev` version
- [ ] Documentation updated if needed

## Next Steps

After your first successful release:

1. Update project README with release badge and download link
2. Configure website to fetch release data
3. Share release with collaborators
4. Plan next release features
5. Monitor for issues and prepare hotfix if needed

## References

- [Full Release Documentation](releases.md)
- [Pipeline Architecture](pipeline.md)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Semantic Versioning](https://semver.org/)
