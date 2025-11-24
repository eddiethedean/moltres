# Release Process

This document formalizes the release process for Moltres.

## Release Checklist

### Pre-Release

- [ ] All tests pass (`pytest -n 10`)
- [ ] Type checking passes (`mypy src`)
- [ ] Linting passes (`ruff check .` and `ruff format --check .`)
- [ ] Documentation is reviewed and updated
- [ ] CHANGELOG.md is updated with all changes
- [ ] Version number is bumped in `pyproject.toml` and `src/moltres/__init__.py`
- [ ] Dependencies are reviewed for security issues
- [ ] Performance benchmarks are within acceptable ranges
- [ ] Breaking changes are documented
- [ ] Migration guide is updated (if needed)

### Release Steps

1. **Update Version**
   ```bash
   # Update pyproject.toml
   version = "X.Y.Z"
   
   # Update src/moltres/__init__.py
   __version__ = "X.Y.Z"
   ```

2. **Update CHANGELOG.md**
   - Move "Unreleased" changes to new version section
   - Add release date
   - Categorize changes (Added, Changed, Fixed, Removed, Security)

3. **Commit Changes**
   ```bash
   git add pyproject.toml src/moltres/__init__.py CHANGELOG.md
   git commit -m "Release X.Y.Z"
   ```

4. **Create Tag**
   ```bash
   git tag vX.Y.Z
   git push origin main
   git push origin vX.Y.Z
   ```

5. **Build Package**
   ```bash
   python -m build
   ```

6. **Test Installation**
   ```bash
   pip install dist/moltres-*.whl
   python -c "import moltres; print(moltres.__version__)"
   ```

7. **Upload to PyPI**
   ```bash
   twine upload dist/*
   ```

8. **Create GitHub Release**
   - Go to GitHub Releases
   - Create new release from tag vX.Y.Z
   - Copy relevant CHANGELOG.md section
   - Publish release

9. **Verify Release**
   ```bash
   pip install --upgrade moltres
   python -c "import moltres; print(moltres.__version__)"
   ```

### Post-Release

- [ ] Monitor PyPI download statistics
- [ ] Monitor GitHub issues for release-related problems
- [ ] Update documentation if needed
- [ ] Announce release (if applicable)

## Version Numbering

Follow [Semantic Versioning](https://semver.org/):

- **MAJOR** (X.0.0): Breaking changes
- **MINOR** (0.X.0): New features (backward compatible)
- **PATCH** (0.0.X): Bug fixes (backward compatible)

### Examples

- `0.11.0` → `0.11.1`: Patch release (bug fix)
- `0.11.0` → `0.12.0`: Minor release (new feature)
- `0.11.0` → `1.0.0`: Major release (breaking change)

## Release Types

### Patch Release (X.Y.Z → X.Y.Z+1)

**When:** Bug fixes, security patches, documentation updates

**Process:**
1. Create branch from main: `git checkout -b release/X.Y.Z+1`
2. Make fixes
3. Update version and CHANGELOG
4. Merge to main
5. Tag and release

### Minor Release (X.Y.Z → X.Y+1.0)

**When:** New features, backward-compatible changes

**Process:**
1. Create branch from main: `git checkout -b release/X.Y+1.0`
2. Add features
3. Update version and CHANGELOG
4. Merge to main
5. Tag and release

### Major Release (X.Y.Z → X+1.0.0)

**When:** Breaking changes

**Process:**
1. Create branch from main: `git checkout -b release/X+1.0.0`
2. Make breaking changes
3. Update version and CHANGELOG
4. Create migration guide
5. Merge to main
6. Tag and release
7. Announce breaking changes

## Release Criteria

### Must Have

- All tests pass
- No critical bugs
- Documentation is up to date
- CHANGELOG is updated

### Should Have

- Performance benchmarks pass
- Security scan passes
- Dependencies are up to date

### Nice to Have

- New features have examples
- Migration guides for breaking changes
- Performance improvements documented

## Rollback Procedure

If a release has critical issues:

1. **Identify Issue**
   - Document the problem
   - Determine severity

2. **Decide on Rollback**
   - Critical bug affecting many users → Rollback
   - Minor issue → Hotfix release

3. **Rollback Steps**
   ```bash
   # Remove PyPI release (if possible)
   # Note: PyPI doesn't allow deletion, but can yank
   twine upload --skip-existing --repository pypi dist/*
   
   # Create hotfix release
   git checkout -b hotfix/X.Y.Z+1
   # Fix issue
   git commit -m "Hotfix: Fix critical issue"
   git tag vX.Y.Z+1
   git push origin hotfix/X.Y.Z+1
   git push origin vX.Y.Z+1
   ```

4. **Communicate**
   - Update GitHub release notes
   - Notify users if needed
   - Document issue and fix

## Automation

### GitHub Actions

The CI workflow automatically:
- Runs tests on all supported platforms
- Checks code quality
- Validates documentation

### Release Automation (Future)

Potential automation:
- Automatic version bumping
- Automatic CHANGELOG generation
- Automatic PyPI upload on tag
- Automatic GitHub release creation

## Release Notes Template

```markdown
## [X.Y.Z] - YYYY-MM-DD

### Added
- New feature 1
- New feature 2

### Changed
- Changed behavior 1
- Changed behavior 2

### Fixed
- Bug fix 1
- Bug fix 2

### Removed
- Removed feature 1 (if any)

### Security
- Security fix 1 (if any)
```

## Approval Process

### Patch Release
- Self-approval for maintainers
- Quick review for non-maintainers

### Minor Release
- Review by at least one other maintainer
- All tests must pass

### Major Release
- Review by all maintainers
- Discussion of breaking changes
- Migration guide required

## Communication

### Release Announcements

- **GitHub Release**: Always create
- **Twitter/X**: For major releases
- **Blog Post**: For major releases or significant features
- **Email**: For security releases (if applicable)

### Release Channels

1. **PyPI**: Primary distribution channel
2. **GitHub Releases**: Release notes and downloads
3. **Documentation**: Update docs site (if applicable)

## Emergency Releases

For critical security issues:

1. **Immediate Action**
   - Create security branch
   - Fix issue
   - Test thoroughly
   - Release immediately

2. **Communication**
   - Security advisory on GitHub
   - Email to users (if applicable)
   - Update documentation

3. **Follow-up**
   - Post-mortem (if needed)
   - Prevent similar issues

