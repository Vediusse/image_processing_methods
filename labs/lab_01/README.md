# ЛР 1. Вычисление яркости на поверхности треугольника

Отдельный проект для первой лабораторной работы по курсу «Методы обработки изображений».

## Что входит

- расчет освещенности и яркости в точках треугольника
- поддержка нескольких точечных источников света
- BRDF с диффузной и зеркальной составляющими
- GUI-приложение на `PySide6`
- 3D-схема сцены
- 2D-рендер поверхности из точки наблюдателя
- CLI-режим
- экспорт результатов
- тесты
- отчет на `Typst`

## Структура

```text
lab_01/
  src/image_lab1/
  tests/
  examples/
  report/
  tools/
  pyproject.toml
```

## Установка и запуск

```bash
cd labs/lab_01
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
image-lab1-gui
```

CLI:

```bash
cd labs/lab_01
image-lab1-cli --scenario examples/demo_scenario.json --export-dir exports
```

Тесты:

```bash
cd labs/lab_01
pytest
```

## Отчет

Генерация артефактов и компиляция отчета:

```bash
cd labs/lab_01
python3 tools/generate_report_assets.py
typst compile report/main.typ report/output/lab_01_report.pdf
```
