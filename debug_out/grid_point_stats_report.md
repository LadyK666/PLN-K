# PLN Grid Point Statistics Report (VOC2007+VOC2012 Trainval)

This document records the mixed `VOC2007 trainval + VOC2012 trainval` statistics for PLN target design.

## Statistical Setup

- Dataset: mixed `VOC2007 trainval` + `VOC2012 trainval`
- Grid setting: `image_size=448`, `grid_size=14` (14x14 cells)
- Point extraction per box: `4 corners + 1 center`
- Overflow threshold parameter: `B=2`
  - Total points per cell overflow: `> 2B = 4`
  - Center points per cell overflow: `> B = 2`
  - Corner points per cell overflow: `> B = 2`
- Split: `trainval`

---

## Mixed VOC2007 + VOC2012 (trainval)

### Global Counts

- Total images: `16551`
- Total boxes: `47223`
- Total points: `236115`
  - Center points: `47223`
  - Corner points: `188892`
- Non-empty grid cells seen: `199952`

### Overflow Grid Counts

- Cells with total points `> 2B`: `1303`
- Cells with center points `> B`: `137`
- Cells with corner points `> B`: `4389`

### Overflow Cell Composition

- Overflow cells total: `4424`
- Single-class overflow cells: `3097` (`70.00%`)
- Multi-class overflow cells: `1327` (`29.99%`)

### Overflow Image Rate

- Images containing overflow cells: `1710`
- Overflow image rate: `10.33%` (`1710/16551`)

### Most Frequent Classes (All Points, Top 20)

1. person (`77880`)
2. chair (`21690`)
3. car (`20040`)
4. bottle (`10580`)
5. dog (`10395`)
6. bird (`9100`)
7. pottedplant (`8620`)
8. cat (`8080`)
9. boat (`6985`)
10. sheep (`6735`)
11. aeroplane (`6425`)
12. sofa (`6055`)
13. bicycle (`6040`)
14. tvmonitor (`5965`)
15. horse (`5780`)
16. motorbike (`5705`)
17. cow (`5290`)
18. diningtable (`5285`)
19. train (`4920`)
20. bus (`4545`)

### Class Ratio in Multi-class Overflow Cells (Top 20)

Definition: `count_in_multiclass_overflow_cells / count_in_all_cells`

1. diningtable: `4.806%` (`254/5285`)
2. bicycle: `4.719%` (`285/6040`)
3. bottle: `3.535%` (`374/10580`)
4. chair: `3.472%` (`753/21690`)
5. motorbike: `3.225%` (`184/5705`)
6. car: `3.214%` (`644/20040`)
7. person: `3.062%` (`2385/77880`)
8. bus: `2.090%` (`95/4545`)
9. horse: `1.834%` (`106/5780`)
10. pottedplant: `1.381%` (`119/8620`)
11. tvmonitor: `1.207%` (`72/5965`)
12. boat: `1.160%` (`81/6985`)
13. cow: `0.813%` (`43/5290`)
14. sofa: `0.727%` (`44/6055`)
15. bird: `0.451%` (`41/9100`)
16. aeroplane: `0.342%` (`22/6425`)
17. train: `0.285%` (`14/4920`)
18. sheep: `0.208%` (`14/6735`)
19. dog: `0.173%` (`18/10395`)
20. cat: `0.025%` (`2/8080`)

---

## Raw Outputs

- `debug_out/grid_point_stats_2007_2012_trainval.json`

These JSON files contain full sorted class lists and overflow examples.
