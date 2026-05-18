#let image_width = "500"
#let image_height = "500"
#let spp_reference = "8"
#let max_depth = "6"
#let gamma_value = "2.20"
#let normalization_mode = "max"
#let triangle_count = "26"
#let light_count = "2"
#let max_radiance = "15.98071406"
#let mean_radiance = "0.44900863"

#let convergence_table = table(
  columns: 3,
  inset: 6pt,
  stroke: (x, y) => if y == 0 { 0.9pt + rgb("#3f5974") } else { 0.5pt + rgb("#8aa0b5") },
  align: center,
  [SPP], [MSE к эталону], [Средняя яркость],
  [1], [7.18164923e-02], [4.49333271e-01],
  [2], [3.74067728e-02], [4.48684497e-01],
  [4], [2.43818644e-02], [4.48546346e-01],
  [6], [1.99635931e-02], [4.48577554e-01],
)
