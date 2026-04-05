#let sample_count_label = "100000"
#let uniformity_rectangle_count = "8"
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
#let triangle_uniformity = "0.006970"

#let circle_inside = "1.000000"
#let circle_constraint = "4.135297e-17"
#let circle_mean = "1.583747e-03"
#let circle_uniformity = "0.010173"

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
  [Треугольник], [1.000000], [1.670430e-17], [3.715492e-03], [0.006970],
  [Круг], [1.000000], [4.135297e-17], [1.583747e-03], [0.010173],
  [Сфера], [1.000000], [2.288503e-17], [3.152223e-03], [0.004246],
  [Косинусное распределение], [1.000000], [1.487255e-17], [1.124648e-04], [0.008468],
)

#let triangle_rectangles_table = table(
  columns: 3,
  inset: 6pt,
  stroke: (x, y) => if y == 0 { 0.9pt + rgb("#3f5974") } else { 0.5pt + rgb("#8aa0b5") },
  align: center,
  [Прямоугольник], [Диапазон в (u, v)], [Число точек],
  [T1], [u∈[0.00; 0.20], v∈[0.00; 0.20]], [8005],
  [T2], [u∈[0.20; 0.40], v∈[0.00; 0.20]], [7999],
  [T3], [u∈[0.60; 0.80], v∈[0.00; 0.20]], [7988],
  [T4], [u∈[0.00; 0.20], v∈[0.20; 0.40]], [8066],
  [T5], [u∈[0.20; 0.40], v∈[0.20; 0.40]], [7959],
  [T6], [u∈[0.40; 0.60], v∈[0.20; 0.40]], [8085],
  [T7], [u∈[0.20; 0.40], v∈[0.40; 0.60]], [7920],
  [T8], [u∈[0.00; 0.20], v∈[0.60; 0.80]], [7927],
)

#let circle_rectangles_table = table(
  columns: 3,
  inset: 6pt,
  stroke: (x, y) => if y == 0 { 0.9pt + rgb("#3f5974") } else { 0.5pt + rgb("#8aa0b5") },
  align: center,
  [Прямоугольник], [Диапазон в локальных (x, y)], [Число точек],
  [C1], [x∈[-0.72; -0.24], y∈[-0.72; -0.24]], [5176],
  [C2], [x∈[-0.24; 0.24], y∈[-0.72; -0.24]], [5029],
  [C3], [x∈[0.24; 0.72], y∈[-0.72; -0.24]], [5111],
  [C4], [x∈[-0.72; -0.24], y∈[-0.24; 0.24]], [5148],
  [C5], [x∈[0.24; 0.72], y∈[-0.24; 0.24]], [5144],
  [C6], [x∈[-0.72; -0.24], y∈[0.24; 0.72]], [5120],
  [C7], [x∈[-0.24; 0.24], y∈[0.24; 0.72]], [5021],
  [C8], [x∈[0.24; 0.72], y∈[0.24; 0.72]], [5096],
)
    