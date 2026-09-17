# Data split registry

No real-data split is frozen yet.

Before using real data, add a machine-readable manifest containing:

- provider and dataset version;
- retrieval timestamp and checksums;
- asset identifiers;
- UTC date boundaries;
- sampling interval and day-coverage threshold;
- development, pilot, and final roles;
- exclusions and their reasons;
- embargo length;
- confirmation that pilot-only assets and windows do not appear in final results.

The final split must be frozen before model checkpoints are evaluated on it.

