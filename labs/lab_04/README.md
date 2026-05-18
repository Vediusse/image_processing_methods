# ЛР 4. Формирование изображения методом трассировки путей

Отдельный проект для четвертой лабораторной работы по курсу «Методы обработки изображений».

Проект реализует path tracing для треугольных сеток с учетом:

- глобального освещения;
- протяженных ламбертовых источников;
- смешанных диффузных и зеркальных материалов;
- выбора события по значимости и русской рулетке;
- антиалиасинга за счет случайного сэмплирования внутри пикселя;
- экспорта изображения в `PPM`.
- интерактивного `Taichi + Metal` realtime-preview для быстрого просмотра сцены.

## Запуск

```bash
cd labs/lab_04
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
image-lab4-gui
```

CLI:

```bash
cd labs/lab_04
image-lab4-cli --config examples/default_scene.json --output outputs/demo.ppm
```

Realtime-preview через GPGPU:

```bash
cd labs/lab_04
image-lab4-cli --config examples/default_scene.json --output outputs/realtime.ppm --png outputs/realtime.png --realtime
```

Проверить скорость нескольких кадров:

```bash
image-lab4-cli --config examples/default_scene.json --output outputs/realtime.ppm --png outputs/realtime.png --realtime --frames 10
```

Первый запуск `--realtime` включает JIT-компиляцию Taichi kernel, поэтому он может быть медленнее. Следующие кадры считаются на `Metal` значительно быстрее.

Progressive Monte-Carlo path tracing через `Taichi + Metal`:

```bash
image-lab4-cli --config examples/default_scene.json --output outputs/gpu_pathtrace.ppm --png outputs/gpu_pathtrace.png --hdr outputs/gpu_pathtrace.hdr --gpu-pathtrace --frames 60
```

В этом режиме каждый кадр добавляет один Monte-Carlo sample per pixel в накопительный буфер. Это честный path tracing с несколькими отскоками, прямым светом от area-light источников, diffuse/mirror событиями и русской рулеткой. Первый кадр включает JIT-компиляцию kernel, поэтому для оценки скорости нужно смотреть `warm FPS`.

## Фильтрация шума

Финальный GPU path tracing фильтруется до tone mapping в линейных единицах яркости. Это важно для физики: фильтр не работает с gamma-corrected PNG, не дорисовывает энергию из воздуха и может сохранять суммарный поток внутри объекта.

Доступные фильтры вынесены в отдельный сервис:

- `none` — без фильтрации;
- `box` / `arithmetic` — арифметическое усреднение;
- `gaussian` — низкочастотный фильтр Гаусса;
- `median` — медианный фильтр с нормировкой энергии по объектам;
- `bilateral` — основной режим для защиты: билатеральная фильтрация по яркости, глубине, нормали и индексу объекта.

Переключение через конфиг `denoise` или CLI:

```bash
image-lab4-cli --config examples/default_scene.json --output outputs/gpu_pathtrace.ppm --png outputs/gpu_pathtrace.png --gpu-pathtrace --frames 20 --filter bilateral --filter-radius 2 --filter-strength 0.95
```

Для сравнения без шумоподавления:

```bash
image-lab4-cli --config examples/default_scene.json --output outputs/gpu_raw.ppm --png outputs/gpu_raw.png --gpu-pathtrace --frames 20 --no-denoise
```
