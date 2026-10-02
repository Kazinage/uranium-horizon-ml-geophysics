# Data contract

Project borehole data are not distributed with this repository.

A compatible table should contain, at minimum:

```text
well_id
depth_m
gamma
resistivity
sp
productive
```

Optional columns may include lithology, original/weak-label provenance, permeability class, radium concentration, profile position and other geological controls.

## Data governance

Do not upload confidential borehole coordinates, assays, resource data or proprietary logs unless you have explicit permission to redistribute them.

The synthetic example is provided so the code can be inspected and executed without project data.
