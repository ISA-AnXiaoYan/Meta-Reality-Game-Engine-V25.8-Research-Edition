# Third-party notices

The first release has no vendored third-party source, hardware SDK, dataset,
or model weight. Runtime dependencies are intentionally empty. Optional
integrations must be documented here before distribution.

The GitHub Actions quality workflow installs `pytest` only in CI; it is not a
runtime or release dependency and is not bundled into the source distribution.
