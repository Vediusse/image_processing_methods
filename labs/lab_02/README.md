# ЛР 2. Расчет интеграла методом Монте-Карло

Отдельный проект для второй лабораторной работы по курсу «Методы обработки изображений».

## Что реализовано

- аналитический расчет интеграла `f(x) = x^2` на `[2, 5]`
- простое интегрирование методом Монте-Карло
- стратифицированное интегрирование с шагами `1.0` и `0.5`
- интегрирование с выборкой по значимости для `p(x) ~ x`, `p(x) ~ x^2`, `p(x) ~ x^3`
- многократная выборка по значимости для `p(x) ~ x` и `p(x) ~ x^3`
- два варианта весов MIS:
  - средняя плотность вероятности
  - средний квадрат плотностей
- интегрирование с использованием русской рулетки
- таблицы результатов, абсолютных и относительных ошибок, оценок погрешности
- GUI с графиками и таблицами
- CLI-режим и экспорт
- тесты

## Структура

```text
lab_02/
  src/image_lab2/
  tests/
  examples/
  report/
  tools/
```

## Запуск

```bash
cd labs/lab_02
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
image-lab2-gui
```

CLI:

```bash
cd labs/lab_02
image-lab2-cli --config examples/default_config.json --export-dir exports
```

Тесты:

```bash
cd labs/lab_02
pytest
```

## Отчет

```bash
cd labs/lab_02
PYTHONPATH=src python3 tools/generate_report_assets.py
typst compile report/main.typ report/output/lab_02_report.pdf
```
