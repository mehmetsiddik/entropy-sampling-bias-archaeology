# Data folder

## Expected file

```
Yaworsky_etal_2020_sdmdata.csv
```

## Source

The published supplementary dataset of

> Yaworsky, P.M., Vernon, K.B., Spangler, J.D., Brewer, S.C. and Codding, B.F.
> (2020) Advancing predictive modeling in archaeology: an evaluation of
> regression and machine learning methods on the Grand Staircase-Escalante
> National Monument. *PLoS ONE* 15(10): e0239424.

archived at the Digital Archaeological Record (tDAR) under
**doi:10.6067/XCV8457626**. Cite that record, not this repository, when using
the data.

## Columns

| Column | Description |
|---|---|
| (index) | record identifier |
| `pa` | 1 if archaeological material has been recorded, 0 for background |
| `east_west_asp` | decomposed east-west component of aspect, unitless, 5 m |
| `north_south_asp` | decomposed north-south component of aspect, unitless, 5 m |
| `slope` | terrain steepness, degrees, 5 m |
| `springs_cd` | cost-distance to springs, minutes, 5 m |
| `streams_cd` | cost-distance to streams, minutes, 5 m |
| `wetlands_cd` | cost-distance to wetlands, minutes, 5 m |
| `wtrshd_size` | catchment area, m^2, 5 m |
| `NPP_mean_00_15` | mean net primary productivity 2000-2015, kg C m^-2 yr^-1, 1 km |
| `PRISM_tmean_30yr_normal_800mM2_annual_asc` | mean annual temperature, 30-year normal, degrees C, 800 m |
| `GDD_corngrowing_dds_2005` | growing degree-days for maize, 30-year average, days yr^-1, 100 m |

Ten predictors, one response. Cost-distance surfaces were generated in the
source study from a DEM-derived friction surface using Tobler's hiking function.

## Expected contents

```
11,814 records
 1,619 presence   (pa = 1)
10,195 background (pa = 0)
```

`src/entropy.load()` checks these counts and raises if they do not match, so a
wrong or truncated file will fail immediately rather than producing quiet
nonsense.

## No coordinates

This dataset contains **no spatial coordinates**. Archaeological site locations
in the United States are withheld under statutory protections against looting,
and the source study was carried out under a Bureau of Land Management cultural
resource use permit.

This is a property of the source data, not an omission here, and it constrains
what the analysis can do: no geographic grid or neighbourhood structure is
possible, so all entropy quantities are computed over covariate space. See the
root README.
