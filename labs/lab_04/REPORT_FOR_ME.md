# ЛР4 — Отчет Для Себя

Этот файл нужен не для преподавателя, а как шпаргалка по проекту: где что лежит, что за что отвечает, что смотреть на защите и где менять параметры.

## Что уже реализовано

- Path tracing для треугольной сцены
- Импорт `OBJ`
- Диффузные и зеркальные материалы
- Выбор события по значимости
- Русская рулетка
- Цветные area lights
- Antialiasing через случайную точку в пикселе
- Экспорт в `PPM`, `PNG`, `HDR`
- GUI и CLI
- Быстрый preview и более качественный финальный рендер
- GPU backend через `torch + mps`
- Realtime preview через `Taichi + Metal`, чтобы быстро крутить/проверять сцену без ожидания path tracing
- Набор низкочастотных фильтров в отдельном сервисе
- Билатеральное шумоподавление финального GPU path tracing до tone mapping

## Что показывать преподавателю

Если спросят “что именно ты сделал”, коротко отвечать так:

1. Реализовал path tracer для сцены из треугольников.
2. Добавил поддержку импорта `OBJ`.
3. Сделал материалы с ламбертовой и зеркальной компонентой.
4. Реализовал выбор события по значимости и русскую рулетку.
5. Сделал протяженные цветные источники света.
6. Добавил GUI и CLI, чтобы можно было менять сцену и рендерить заново.
7. Добавил экспорт не только в `PPM`, но и в `HDR`.
8. Добавил физически корректную фильтрацию шума в линейной яркости: `box`, `gaussian`, `median`, `bilateral`, переключение через CLI/JSON.

## Где что лежит

### Основной рендер

Файл:
[path_tracer.py](/Users/rublev/DEV/image/image_processing_methods/labs/lab_04/src/image_lab4/services/path_tracer.py)

Смотреть тут:
- `render(...)` — главный вход в рендер
- `_render_torch(...)` — быстрый батчевый рендер через `torch`
- `_intersect_rays_torch(...)` — пересечение лучей с треугольниками
- `_sample_direct_lighting_torch(...)` — прямое освещение от area lights
- `_sample_events_torch(...)` — выбор диффузного или зеркального события
- `_postprocess_radiance(...)` — подавление fireflies и денойз
- `_tone_map(...)` — нормировка и гамма

### Быстрый realtime-preview

Файл:
[taichi_realtime.py](/Users/rublev/DEV/image/image_processing_methods/labs/lab_04/src/image_lab4/services/taichi_realtime.py)

Смысл:
- это отдельный `Taichi + Metal` GPGPU backend;
- он выпускает по одному лучу из камеры на пиксель;
- пересечение со всеми треугольниками и простой свет от area-light источников считаются на GPU;
- режим нужен для интерактивного просмотра сцены, а не для финального физически точного кадра.

Команда:

```bash
image-lab4-cli --config examples/default_scene.json --output outputs/realtime.ppm --png outputs/realtime.png --realtime
```

Для замера скорости:

```bash
image-lab4-cli --config examples/default_scene.json --output outputs/realtime.ppm --png outputs/realtime.png --realtime --frames 10
```

Первый кадр компилирует Taichi kernel, следующие кадры идут быстро. В GUI кнопка `Realtime старт` запускает непрерывный таймерный режим, повторное нажатие останавливает его.

### Progressive Monte-Carlo path tracing на GPU

Файл:
[taichi_progressive_path_tracer.py](/Users/rublev/DEV/image/image_processing_methods/labs/lab_04/src/image_lab4/services/taichi_progressive_path_tracer.py)

Смысл:
- это уже не raster preview, а Monte-Carlo path tracing;
- каждый frame добавляет `1 spp` в накопление;
- внутри есть несколько отскоков, diffuse/mirror sampling, прямой свет от area-light источников и русская рулетка;
- на текущей сцене `500x500`, `max_depth=4`, `60` кадров дали warm FPS около `39`.

Команда:

```bash
image-lab4-cli --config examples/default_scene.json --output outputs/gpu_pathtrace.ppm --png outputs/gpu_pathtrace.png --hdr outputs/gpu_pathtrace.hdr --gpu-pathtrace --frames 60
```

Это главный режим, если нужно объяснять преподавателю именно GPGPU Monte-Carlo path tracing на Mac.

### Фильтры шума

Файл:
[image_filters.py](/Users/rublev/DEV/image/image_processing_methods/labs/lab_04/src/image_lab4/services/image_filters.py)

Что важно:
- фильтры применяются к radiance до tone mapping и gamma correction;
- `box` / `arithmetic` и `gaussian` — классические низкочастотные фильтры из лекции;
- `median` — нелинейный медианный фильтр, после него включена нормировка суммарной энергии по объектам;
- `bilateral` — главный фильтр для финального изображения, потому что он сохраняет границы объектов;
- guide-данные для GPU-рендера: глубина, индекс первого пересеченного треугольника и нормаль поверхности;
- прямая яркость первого попадания фильтруется отдельно от вторичной яркости, чтобы не смешивать разные физические компоненты освещения.

Команды:

```bash
image-lab4-cli --config examples/default_scene.json --output outputs/gpu_filtered.ppm --png outputs/gpu_filtered.png --gpu-pathtrace --frames 20 --filter bilateral --filter-radius 2 --filter-strength 0.85
image-lab4-cli --config examples/default_scene.json --output outputs/gpu_raw.ppm --png outputs/gpu_raw.png --gpu-pathtrace --frames 20 --no-denoise
```

### Структуры сцены

Файл:
[scene.py](/Users/rublev/DEV/image/image_processing_methods/labs/lab_04/src/image_lab4/models/scene.py)

Смотреть тут:
- `Material`
- `Triangle`
- `Camera`
- `RenderSettings`
- `SceneConfig`
- `Scene`
- `RenderArtifact`

### Векторы и цвет

Файл:
[vector.py](/Users/rublev/DEV/image/image_processing_methods/labs/lab_04/src/image_lab4/models/vector.py)

Тут базовая математика:
- `Vec3`
- `Point3`
- `ColorRGB`

### Геометрия

Файл:
[geometry.py](/Users/rublev/DEV/image/image_processing_methods/labs/lab_04/src/image_lab4/math/geometry.py)

Что здесь:
- отражение
- cosine hemisphere sampling
- нормаль треугольника
- площадь треугольника
- выбор точки на треугольнике

### Загрузка сцены из JSON

Файл:
[config_loader.py](/Users/rublev/DEV/image/image_processing_methods/labs/lab_04/src/image_lab4/io/config_loader.py)

Здесь:
- `load_config(...)`
- `load_config_from_text(...)`
- `save_config(...)`

Если нужно добавить новый параметр сцены, чаще всего менять надо именно тут и в `models/scene.py`.

### Импорт OBJ

Файл:
[obj_loader.py](/Users/rublev/DEV/image/image_processing_methods/labs/lab_04/src/image_lab4/io/obj_loader.py)

Если спросят, как работает OBJ:
- читаются строки `v`
- читаются строки `f`
- многоугольники триангулируются “веером”

### GUI

Файл:
[main_window.py](/Users/rublev/DEV/image/image_processing_methods/labs/lab_04/src/image_lab4/ui/main_window.py)

Что важно:
- `Быстрый превью` — урезанный быстрый рендер
- `Финальный рендер` — обычный режим
- тут же кнопки `Сохранить PPM / PNG / HDR`

### CLI

Файл:
[cli.py](/Users/rublev/DEV/image/image_processing_methods/labs/lab_04/src/image_lab4/cli.py)

Полезно для быстрого запуска без GUI.

### Экспорт файлов

Файл:
[exporters.py](/Users/rublev/DEV/image/image_processing_methods/labs/lab_04/src/image_lab4/report/exporters.py)

Тут:
- `save_ppm(...)`
- `save_png(...)`
- `save_hdr(...)`

### Генерация отчета

Файл:
[generate_report_assets.py](/Users/rublev/DEV/image/image_processing_methods/labs/lab_04/tools/generate_report_assets.py)

Тут собираются:
- `final_render.png`
- `final_render.ppm`
- `final_render.hdr`
- `scene_overview.png`
- `convergence.png`
- `values.typ`

## Где менять сцену

Основной файл сцены:
[default_scene.json](/Users/rublev/DEV/image/image_processing_methods/labs/lab_04/examples/default_scene.json)

Если нужно поменять:
- камеру
- число лучей
- глубину пути
- материалы
- треугольники
- источники света
- OBJ-модели

то почти всё меняется именно здесь.

OBJ-пример:
[obj_scene.json](/Users/rublev/DEV/image/image_processing_methods/labs/lab_04/examples/obj_scene.json)

OBJ-файл:
[pyramid.obj](/Users/rublev/DEV/image/image_processing_methods/labs/lab_04/examples/pyramid.obj)

## Что именно мы используем из теории

### 1. Path tracing

Смысл:
- интеграл освещенности оценивается Монте-Карло выборкой случайных путей

У нас это проявляется так:
- из камеры летит луч
- он пересекает сцену
- на поверхности выбирается новое направление
- путь продолжается до остановки

### 2. Ламберт

Для диффузной части:
- отражение одинаково по всем направлениям полусферы
- новое направление берется cosine-weighted sampling

### 3. Зеркальное отражение

Для зеркальной части:
- направление строится как идеальное отражение относительно нормали

### 4. Выбор события по значимости

Если у материала есть и diffuse, и mirror:
- выбирается одно событие
- вероятность пропорциональна энергии компоненты

### 5. Русская рулетка

После нескольких первых bounce:
- путь может быть случайно остановлен
- если не остановлен, throughput пересчитывается

### 6. Протяженные источники света

Свет не точечный:
- источник — это светящийся треугольник
- на нем случайно выбирается точка

### 7. Tone mapping

После расчета абсолютных яркостей:
- значения нормируются
- clamp до `[0, 1]`
- gamma correction

## Если преподаватель спросит “почему было шумно”

Отвечать так:

- path tracing — метод Монте-Карло, шум для малого числа лучей естественен
- маленькие цветные источники света увеличивают дисперсию
- для этого был добавлен более быстрый GPU backend, preview-режим и постобработка против fireflies

## Если нужно что-то поменять

### Сделать сцену другой

Менять:
[default_scene.json](/Users/rublev/DEV/image/image_processing_methods/labs/lab_04/examples/default_scene.json)

### Изменить материалы

Менять:
[default_scene.json](/Users/rublev/DEV/image/image_processing_methods/labs/lab_04/examples/default_scene.json)

Блок:
`materials`

### Изменить источники света

Менять:
[default_scene.json](/Users/rublev/DEV/image/image_processing_methods/labs/lab_04/examples/default_scene.json)

Это обычные треугольники с полем:
`"emission": [...]`

### Изменить качество / скорость

Менять:
[default_scene.json](/Users/rublev/DEV/image/image_processing_methods/labs/lab_04/examples/default_scene.json)

Блок `render`:
- `samples_per_pixel`
- `max_depth`
- `min_depth`
- `normalization`
- `normalization_value`

### Изменить preview-режим

Менять:
[main_window.py](/Users/rublev/DEV/image/image_processing_methods/labs/lab_04/src/image_lab4/ui/main_window.py)

и
[cli.py](/Users/rublev/DEV/image/image_processing_methods/labs/lab_04/src/image_lab4/cli.py)

Там сейчас preview понижает:
- разрешение
- `SPP`
- глубину пути

### Изменить denoise / firefly suppression

Менять:
[path_tracer.py](/Users/rublev/DEV/image/image_processing_methods/labs/lab_04/src/image_lab4/services/path_tracer.py)

Методы:
- `_postprocess_radiance(...)`
- `_edge_aware_denoise(...)`

### Изменить HDR export

Менять:
[exporters.py](/Users/rublev/DEV/image/image_processing_methods/labs/lab_04/src/image_lab4/report/exporters.py)

## Какие команды полезно помнить

### Быстрый preview

```bash
cd /Users/rublev/DEV/image/image_processing_methods/labs/lab_04
PYTHONPATH=src python3 -m image_lab4.cli --config examples/default_scene.json --output report/generated/preview.ppm --png report/generated/preview.png --hdr report/generated/preview.hdr --preview
```

### Финальный рендер

```bash
cd /Users/rublev/DEV/image/image_processing_methods/labs/lab_04
PYTHONPATH=src python3 -m image_lab4.cli --config examples/default_scene.json --output report/generated/final.ppm --png report/generated/final.png --hdr report/generated/final.hdr
```

### GUI

```bash
cd /Users/rublev/DEV/image/image_processing_methods/labs/lab_04
PYTHONPATH=src python3 -m image_lab4.app
```

### Тесты

```bash
cd /Users/rublev/DEV/image/image_processing_methods/labs/lab_04
python3 -m pytest tests/test_path_tracer.py tests/test_obj_loader.py
```

## Как коротко объяснить проект на защите

“Я сделал path tracer для сцен из треугольников. Сцена задается вручную и через OBJ. Материалы поддерживают диффузную и зеркальную составляющие, событие выбирается по значимости, длина пути ограничивается русской рулеткой. Источники света протяженные, цветные, ламбертовы. Есть GUI, CLI, экспорт в PPM и HDR, а для ускорения я использовал torch с backend MPS на Apple Silicon.”
