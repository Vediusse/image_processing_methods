#let sample_count_label = "100000"
#let triangle_a = "(0.000, 0.000, 0.000)"
#let triangle_b = "(2.000, 0.000, 0.000)"
#let triangle_c = "(0.500, 1.800, 0.700)"
#let circle_center = "(-1.000, 0.500, 0.800)"
#let circle_normal = "(0.200, -0.300, 1.000)"
#let circle_radius = "1.200"
#let cosine_normal = "(0.000, 0.000, 1.000)"

#let triangle_inside = "1.000000"
#let triangle_constraint = "1.670430e-17"
#let triangle_mean = "3.715492e-03"
#let triangle_uniformity = "0.012086"

#let circle_inside = "1.000000"
#let circle_constraint = "4.135297e-17"
#let circle_mean = "1.583747e-03"
#let circle_uniformity = "0.014896"

#let sphere_inside = "1.000000"
#let sphere_constraint = "2.288503e-17"
#let sphere_mean = "3.152223e-03"
#let sphere_uniformity = "0.004246"

#let cosine_inside = "1.000000"
#let cosine_constraint = "1.487255e-17"
#let cosine_mean = "1.124648e-04"
#let cosine_uniformity = "0.008468"

#let metrics_table = table(
  columns: 5,
  inset: 6pt,
  stroke: (x, y) => if y == 0 { 0.9pt + rgb("#3f5974") } else { 0.5pt + rgb("#8aa0b5") },
  align: center,
  [Распределение], [Inside ratio], [Constraint error], [Mean error], [Uniformity score],
  [Треугольник], [1.000000], [1.670430e-17], [3.715492e-03], [0.012086],
  [Круг], [1.000000], [4.135297e-17], [1.583747e-03], [0.014896],
  [Сфера], [1.000000], [2.288503e-17], [3.152223e-03], [0.004246],
  [Косинусное распределение], [1.000000], [1.487255e-17], [1.124648e-04], [0.008468],
)
