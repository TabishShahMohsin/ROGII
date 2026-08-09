# ROGII

![Problem Diagram](Problem%20Diagram.png)

* Modern drilling is a remote-controlled, blindfolded 3D underground navigation. Modern rigs use steerable drill bits.
* A "horizontal well" starts by drilling vertically, but then gets curved and is maintained horizontally into the most oil yeilding layer. Keeping the TVT constant. (Z may change)
* Hence an **L-Profile.**, also called as a **Horizontal Well**. [Video on Profiles.](https://www.youtube.com/watch?v=Xr_kSHJguTM&t=167s)
* Data is such that the layers including the a surface are exactly **parallel surfaces**.
    * Almost parallel planes.
* `Purpose`: Maintaining TVT to be constant. (Geosteering)
* `Objective`: Making a model for predicting current TVT.
* The Drill Bit has a **Gamma Ray (GR)** tool/sensor. Different rocks naturally emit different amounts of radiation (Gamma Rays).
    * For example, "Shale" (which holds oil) emits high gamma rays, while "Limestone" emits low gamma rays.
* A type well csv, is basically a lookup where `GR` vs `TVT` is mapped for a place.
    * As found later, many wells share the sub sequence of a common type well.
* Everything meassured from sea level is -ve.

Provided:
* Type Well: A master map of gamma-ray signals vs TVT.
* `GR`- Gamma Ray : Gamma Ray signal values sensed from the drill bit.
* `WELLNAME` - Unique identifier for the well.
* `MD` - Measured Depth (ft): Total length of all the pipe currently in the hole (Curve's Perimeter).
* `X`, `Y`, `Z` - Treat them as standard 3D Cartesian coordinates (ft). (Global Reference)
* `ANCC`, `ASTNU`, `ASTNL`, `EGFDU`, `EGFDL`, `BUDA` - Predicted depth of different types of layers.
    * For the XYZ at what depth is the depth of the directly above `ANCC`: Check the value of `ANCC` for that row.
    * Only in the training data.
* `TVT_input` - Some initial values of TVT.

To Predict:
* `TVT`: True Vertical Thickness
    * Vertical: Along Gravity.
    * A loosely used term in the oil industry. Doesn't mean [this.](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQx0VWc2QpBTAH6y6KrZlkB9v8hVtXKXfd29w&s)
    * Here, it means the depth of the drill bit from the ground surface.
    * Know that ground surface isn't perpendicular to Z.


Sources:
* [ROGII-Software Guide](https://rogii.com/products/starsteer#:~:text=Compare%20wells%20and%20isopachs%20using,Subsurface%20Geological%20Mapping)
* [Kaggle's ROGII - Wellbore Geology Prediction Competition](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction)

## 🗺 Well Map
(python /scripts/map_viewer.py <well_name>)

A program for XY well plot, analyzing 3D proximity and parallelism, and projecting neighbor TVT mappings with RMSE metrics.

## 🧭 Geosteering Simulator of a Well 
(python /scripts/well_viewer.py <well_name>)

A geosteering program with GR correlation, noise filtering, chunk-based anchoring, and multi-axis view support (TVD, TVT, and stratigraphic offset).

## Weird wells:
* Geologist Mistakes:
    * d7eb0be8
* At least 3 paths not found in phys pipeline
    * fba7683c
* Closest start point to its start is about 18000 m away.
    * 059c8f24

# Anamolies 
| Well Names | Reason | Remark |
| -- | -- | -- |
| 059c8f24, 14ab73fb, eba6605e | Δ TVT is abrupt | Single row in each TW|
| 4a335117 | RMSE of Linear Regression | RMSE = 36, 2nd hightest = 24 |
| 1b1eba53, d7eb0be8, 81bf5923, 4c2208f5, 03a935ae, 727a3a10, a8ed028a | Empty ANCC | Whole column in HW
| 9dfff011 | Empty EGFDL | Whole Column in HW
| 03a935ae, 1b1eba53, 4c2208f5, 4f3eb9e9, 727a3a10, 78a4a386, 81bf5923, a8ed028a, d7eb0be8, 5d7198fd, 6e9ccd38 | Missing ANCC | in TW |
| 86454a6f | Missing ANCC, ASTNU | in TW |
| 0bbf5e67, 2cee0cba, 353e5502, 4f4ac5ce, 5aef5c6c, 5eae34a8, 99529c45, a87433c9, d60430e6, d00e7eb9, d90aa14c | Missing BUDA | in TW |
| 761 out of 773 Wells | Different than 6 layers | in TW