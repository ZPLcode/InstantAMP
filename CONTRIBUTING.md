# Contributing

InstantAMP maintains one walk, run and recovery task per articulation. Put new Zenbot joint names, limits and order mappings in the model definition. Apply changes to commands, rewards, motion sampling or policy updates to the shared code and exercise both joint counts. Include a regression test for a bug that behaves differently on the two models.

Keep each change focused on one reviewable outcome. Use a short imperative commit subject, followed by a blank line and the reason for the change. In the pull request, describe the user-visible effect, the tests you ran and any external data or hardware needed to reproduce a result. [Google engineering review practices](https://google.github.io/eng-practices/review/developer/small-cls.html) · [Commit descriptions](https://google.github.io/eng-practices/review/developer/cl-descriptions.html) · [Google Python style guide](https://google.github.io/styleguide/pyguide.html)

Install Ruff for development, then check the imports, Python style, CPU tests and patch before submitting.

```bash
python -m pip install ruff
ruff check instantamp tools tests
JAX_PLATFORMS=cpu python -m pytest -q tests
git diff --check
git diff --cached --check
```

Motion datasets, checkpoints, generated meshes and training logs belong in the local data, asset or run directories. Include provenance and license notices with new source assets.
