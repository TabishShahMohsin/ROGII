# ROGII

![Problem Diagram](Problem%20Diagram.png)
*Original source: [zacchaeus](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/697418)*

Checkout the scripts for what I was able to break the problem to.

* Modern drilling is a remote-controlled, blindfolded 3D underground navigation. Modern rigs use steerable drill bits.
* A "horizontal well" starts by drilling vertically, but then gets curved and is maintained horizontally into the most oil yeilding layer. Keeping the TVT constant. (Z may change)
* Hence an **L-Profile.**, also called as a **Horizontal Well**. [Video on Profiles.](https://www.youtube.com/watch?v=Xr_kSHJguTM&t=167s)
* Data is such that the layers including the ground surface are exactly **parallel surfaces**.
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


Citations:
* [ROGGI-Software Guide](https://rogii.com/products/starsteer#:~:text=Compare%20wells%20and%20isopachs%20using,Subsurface%20Geological%20Mapping)

## 🗺 Well Map
(python /scripts/map_viewer.py <well_name>)

A program for XY well plot, analyzing 3D proximity and parallelism, and projecting neighbor TVT mappings with RMSE metrics.

## 🧭 Geosteering Simulator of a Well 
(python /scripts/well_viewer.py <well_name>)

A geosteering program with GR correlation, noise filtering, chunk-based anchoring, and multi-axis view support (TVD, TVT, and stratigraphic offset).

Ideas for scripts:
* Showing GR from the known hw
* Showing GR from near by wells
* Adding GR baseline shift
* Should be able to turn on and off prediction from each of the wells

## Weird wells:
* Geo's Mistakes:
    * d7eb0be8
* 3 paths not found in phys pipeline
    * fba7683c
* Closest start point to its start is about 18000 m away.
    * 059c8f24