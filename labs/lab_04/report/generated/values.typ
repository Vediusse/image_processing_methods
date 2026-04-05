#let image_width = "500"
#let image_height = "500"
#let spp_reference = "12"
#let max_depth = "4"
#let gamma_value = "2.20"
#let normalization_mode = "percentile"
#let triangle_count = "60"
#let light_count = "8"
#let max_radiance = "0.00000000"
#let mean_radiance = "0.00000000"

#let convergence_table = table(
  columns: 3,
  inset: 6pt,
  stroke: (x, y) => if y == 0 { 0.9pt + rgb("#3f5974") } else { 0.5pt + rgb("#8aa0b5") },
  align: center,
  [SPP], [MSE к эталону], [Средняя яркость],

)
