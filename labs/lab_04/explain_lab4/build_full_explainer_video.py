from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parent / "video"
OUT_MP4 = OUT_DIR / "lab4_path_tracing_full_explainer.mp4"
WIDTH = 1280
HEIGHT = 720
FPS = 24

FONT_REGULAR = "/Users/rublev/DEV/image/image_processing_methods/labs/lab_04/.venv/lib/python3.8/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf"
FONT_BOLD = "/Users/rublev/DEV/image/image_processing_methods/labs/lab_04/.venv/lib/python3.8/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans-Bold.ttf"
FONT_MONO = "/Users/rublev/DEV/image/image_processing_methods/labs/lab_04/.venv/lib/python3.8/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSansMono.ttf"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames_dir = OUT_DIR / "frames"
    if frames_dir.exists():
        for old in frames_dir.glob("*.png"):
            old.unlink()
    else:
        frames_dir.mkdir(parents=True)

    renderer = VideoRenderer(frames_dir)
    renderer.render()
    build_mp4(frames_dir)
    print(f"video: {OUT_MP4}")


class VideoRenderer:
    def __init__(self, frames_dir: Path) -> None:
        self.frames_dir = frames_dir
        self.frame_index = 0
        self.title = font(42, bold=True)
        self.h2 = font(29, bold=True)
        self.text = font(23)
        self.small = font(18)
        self.tiny = font(15)
        self.code = font(17, mono=True)
        self.code_big = font(21, mono=True)

    def render(self) -> None:
        segments = [
            (3.2, self.scene_title),
            (4.0, self.scene_stack),
            (4.4, self.scene_config_to_models),
            (4.8, self.scene_world),
            (5.0, self.scene_camera_ray),
            (5.0, self.scene_intersection),
            (5.0, self.scene_material),
            (5.0, self.scene_light_importance),
            (5.0, self.scene_area_and_point_math),
            (4.6, self.scene_shadow_ray),
            (5.0, self.scene_bounce_path),
            (5.2, self.scene_pixel_buffers),
            (5.2, self.scene_spp_accumulation),
            (4.8, self.scene_tone_hdr),
            (4.6, self.scene_gui_cli),
            (5.0, self.scene_code_map),
            (4.2, self.scene_final_script),
        ]
        for duration, draw_func in segments:
            total = max(1, int(duration * FPS))
            for i in range(total):
                t = i / max(total - 1, 1)
                img = self.base()
                draw_func(img, t)
                self.save(img)

    def base(self) -> Image.Image:
        img = Image.new("RGB", (WIDTH, HEIGHT), "#0d1420")
        draw = ImageDraw.Draw(img)
        for y in range(HEIGHT):
            k = y / HEIGHT
            r = int(13 + 8 * k)
            g = int(20 + 16 * k)
            b = int(32 + 24 * k)
            draw.line([(0, y), (WIDTH, y)], fill=(r, g, b))
        draw.rounded_rectangle((24, 24, WIDTH - 24, HEIGHT - 24), 28, outline="#233a55", width=2)
        return img

    def save(self, img: Image.Image) -> None:
        img.save(self.frames_dir / f"frame_{self.frame_index:05d}.png")
        self.frame_index += 1

    def scene_title(self, img: Image.Image, t: float) -> None:
        d = ImageDraw.Draw(img)
        title = "ЛР 4: как работает Monte-Carlo Path Tracing"
        subtitle = "Один ролик: от JSON-сцены до цвета конкретного пикселя"
        center_text(d, title, self.title, 92, "#f6efe3")
        center_text(d, subtitle, self.h2, 156, "#b9d7ff")
        self.draw_simple_scene(d, t, highlight="all")
        chips = ["Python", "PySide6 GUI", "Taichi + Metal GPU", "Monte Carlo", "HDR/PNG"]
        x = 180
        for chip in chips:
            pill(d, (x, 596), chip, self.small, "#17324a", "#6eb6ff")
            x += 190
        center_text(d, "Идея: каждый пиксель = среднее многих случайных световых путей", self.text, 648, "#f8d17a")

    def scene_stack(self, img: Image.Image, t: float) -> None:
        d = ImageDraw.Draw(img)
        header(d, "1. Какие технологии используются")
        items = [
            ("JSON", "описание камеры, материалов, треугольников, area lights и point lights"),
            ("OBJ", "дополнительные сетки: например пирамиды загружаются как треугольники"),
            ("Python-модели", "SceneConfig, Triangle, Material, PointLight, RenderArtifact"),
            ("Taichi + Metal", "GPU kernel считает один случайный path tracing frame"),
            ("PySide6", "GUI показывает progressive accumulation и сохраняет текущий кадр"),
            ("HDR/PNG", "HDR хранит физическую radiance, PNG хранит tone-mapped картинку"),
        ]
        y = 132
        for idx, (left, right) in enumerate(items):
            alpha = ease_step(t, idx / len(items), (idx + 1.4) / len(items))
            if alpha <= 0:
                continue
            card(d, 100, y, 1080, 66, "#111f30", "#2d5378")
            d.text((130, y + 18), left, font=self.h2, fill=blend("#0d1420", "#f8d17a", alpha))
            d.text((290, y + 21), right, font=self.small, fill=blend("#0d1420", "#e6eef9", alpha))
            y += 78

    def scene_config_to_models(self, img: Image.Image, t: float) -> None:
        d = ImageDraw.Draw(img)
        header(d, "2. Сначала JSON превращается в сцену")
        code = '''"materials": [
  { "name": "mirror",
    "diffuse": [0.02, 0.05, 0.02],
    "mirror":  [0.16, 0.78, 0.24] }
],
"point_lights": [
  { "position": [-0.38, 1.15, 0.18],
    "intensity": [3.2, 2.7, 1.7] }
]'''
        code_box(d, 70, 130, 520, 390, code, self.code, title="examples/default_scene.json")
        code2 = """load_config_from_text(...)
  -> Camera
  -> Material[]
  -> Triangle[]
  -> PointLight[]

PathTracer._build_scene(...)
  -> normals
  -> areas
  -> light_probabilities
  -> arrays for GPU"""
        code_box(d, 690, 130, 500, 390, code2, self.code_big, title="Python objects")
        arrow(d, (600, 320), (680, 320), "#f8d17a", width=5)
        d.text((410, 566), "Важно: рендерер работает не с красивыми объектами, а с массивами чисел.", font=self.text, fill="#f6efe3")

    def scene_world(self, img: Image.Image, t: float) -> None:
        d = ImageDraw.Draw(img)
        header(d, "3. Упрощенная сцена для объяснения")
        self.draw_simple_scene(d, t, highlight="all")
        notes = [
            ("diffuse cube", "матовый объект: свет рассеивается"),
            ("green mirror", "зеркальный объект: луч отражается"),
            ("area lights A1/A2", "протяженные источники с площадью"),
            ("point light P1", "точечный источник без площади, но с intensity"),
        ]
        y = 150
        for title, body in notes:
            pill(d, (820, y), title, self.small, "#17324a", "#f8d17a")
            d.text((820, y + 38), body, font=self.small, fill="#dbe9ff")
            y += 112

    def scene_camera_ray(self, img: Image.Image, t: float) -> None:
        d = ImageDraw.Draw(img)
        header(d, "4. Выбираем конкретный пиксель и выпускаем camera ray")
        self.draw_simple_scene(d, t, highlight="camera")
        px, py = (240, 310)
        sx, sy = (475, 360)
        d.rectangle((170, 160, 330, 430), outline="#89c2ff", width=3)
        d.rectangle((220, 290, 260, 330), outline="#f8d17a", width=4)
        d.text((168, 128), "экран камеры", font=self.small, fill="#dbe9ff")
        arrow(d, (px, py), (sx, sy), "#f8d17a", width=5)
        formula = """sample_x = (x + random()) / width
sample_y = (y + random()) / height

ray.direction =
normalize(forward + right*ndc_x + up*ndc_y)"""
        code_box(d, 690, 150, 500, 260, formula, self.code, title="camera ray")
        d.text((710, 452), "random() нужен, чтобы не стрелять всегда в центр пикселя.\nТак появляется Monte Carlo sampling.", font=self.small, fill="#f6efe3")

    def scene_intersection(self, img: Image.Image, t: float) -> None:
        d = ImageDraw.Draw(img)
        header(d, "5. Луч ищет ближайший треугольник")
        self.draw_simple_scene(d, t, highlight="intersection")
        start = (230, 500)
        end = lerp_point(start, (545, 390), t)
        arrow(d, start, end, "#ff6b57", width=5)
        d.ellipse((535, 380, 555, 400), fill="#f8d17a")
        formula = """Moller-Trumbore:
edge1 = b - a
edge2 = c - a
pvec = cross(ray, edge2)
det  = dot(edge1, pvec)

если 0 <= u, 0 <= v, u+v <= 1
и t > 0 -> луч попал в треугольник"""
        code_box(d, 690, 135, 500, 360, formula, self.code, title="intersection")
        d.text((700, 535), "Из всех попаданий выбираем минимальный t: это ближайшая поверхность.", font=self.small, fill="#f6efe3")

    def scene_material(self, img: Image.Image, t: float) -> None:
        d = ImageDraw.Draw(img)
        header(d, "6. Материал решает: diffuse или mirror")
        self.draw_simple_scene(d, t, highlight="material")
        x0, y0 = 690, 130
        code = """"diffuse": [0.02, 0.05, 0.02],
"mirror":  [0.16, 0.78, 0.24]

diffuse[channel] + mirror[channel] <= 1"""
        code_box(d, x0, y0, 500, 170, code, self.code, title="green mirror material")
        d.text((700, 335), "diffuse", font=self.h2, fill="#f8d17a")
        arrow(d, (820, 350), (930, 410), "#f8d17a", width=4)
        d.text((958, 397), "случайный отскок\nпо полусфере", font=self.small, fill="#dbe9ff")
        d.text((700, 465), "mirror", font=self.h2, fill="#8dff9f")
        arrow(d, (820, 480), (940, 480), "#8dff9f", width=4)
        d.text((958, 460), "идеальное отражение:\nreflect(ray, normal)", font=self.small, fill="#dbe9ff")

    def scene_light_importance(self, img: Image.Image, t: float) -> None:
        d = ImageDraw.Draw(img)
        header(d, "7. Источник света выбирается по значимости")
        labels = ["A1", "A2", "P1"]
        weights = [0.9, 0.55, 2.2]
        total = sum(weights)
        probs = [w / total for w in weights]
        x = 150
        for i, (label, p) in enumerate(zip(labels, probs)):
            h = int(330 * p / max(probs))
            color = "#f8d17a" if label.startswith("A") else "#ff9f43"
            d.rounded_rectangle((x, 500 - h, x + 150, 500), 12, fill=color, outline="#f6efe3", width=2)
            d.text((x + 48, 520), label, font=self.h2, fill="#f6efe3")
            d.text((x + 36, 465 - h), f"p={p:.2f}", font=self.small, fill="#f6efe3")
            x += 210
        formula = """Area light:
weight = area * average(emission)

Point light:
weight = average(intensity)

p_i = weight_i / sum(weight)

Вклад потом делится на p_i,
поэтому оценка остается честной."""
        code_box(d, 760, 140, 430, 380, formula, self.code, title="importance sampling")
        d.text((135, 606), "Зачем так? Чтобы чаще стрелять shadow ray к источникам, которые реально сильнее влияют на картинку.", font=self.small, fill="#dbe9ff")

    def scene_area_and_point_math(self, img: Image.Image, t: float) -> None:
        d = ImageDraw.Draw(img)
        header(d, "8. Формула вклада света")
        left = """Area light:
L += throughput * emission * bsdf *
     cos_surface * cos_light
     / (distance² * pdf)

pdf = p_light * (1 / area)"""
        right = """Point light:
L += throughput * intensity * bsdf *
     cos_surface
     / (distance² * p_light)

у point light нет area,
но есть intensity"""
        code_box(d, 80, 150, 535, 360, left, self.code_big, title="протяженный источник")
        code_box(d, 665, 150, 535, 360, right, self.code_big, title="точечный источник")
        d.text((148, 565), "bsdf = diffuse / π для ламбертовой поверхности", font=self.text, fill="#f8d17a")
        d.text((148, 612), "distance² делает дальние источники слабее", font=self.text, fill="#dbe9ff")

    def scene_shadow_ray(self, img: Image.Image, t: float) -> None:
        d = ImageDraw.Draw(img)
        header(d, "9. Shadow ray проверяет, виден ли источник")
        self.draw_simple_scene(d, t, highlight="shadow")
        hit = (555, 390)
        light = (730, 190)
        blocker = (650, 300)
        arrow(d, hit, lerp_point(hit, light, t), "#f8d17a", width=4)
        d.ellipse((blocker[0] - 20, blocker[1] - 20, blocker[0] + 20, blocker[1] + 20), fill="#ff6b57")
        d.text((760, 165), "источник", font=self.small, fill="#f8d17a")
        d.text((680, 284), "если объект перекрыл путь,\nпрямой вклад = 0", font=self.small, fill="#ffb3a7")
        code = """shadow_origin = hit_pos + light_dir * eps
shadow_hit = intersect(shadow_ray)

если shadow_hit раньше источника:
    contribution = 0
иначе:
    добавляем вклад света"""
        code_box(d, 760, 395, 430, 230, code, self.code, title="visibility")

    def scene_bounce_path(self, img: Image.Image, t: float) -> None:
        d = ImageDraw.Draw(img)
        header(d, "10. Путь продолжается несколько отскоков")
        self.draw_simple_scene(d, t, highlight="path")
        points = [(230, 500), (540, 390), (680, 260), (830, 430), (1000, 300)]
        max_seg = (len(points) - 1) * t
        for i in range(len(points) - 1):
            local = min(max(max_seg - i, 0), 1)
            if local > 0:
                arrow(d, points[i], lerp_point(points[i], points[i + 1], local), ["#ff6b57", "#8dff9f", "#f8d17a", "#89c2ff"][i], width=4)
        formula = """throughput *= material_weight / event_pdf

diffuse: cosine-weighted hemisphere
mirror: reflect(direction, normal)

после min_depth:
Russian roulette может остановить путь"""
        code_box(d, 720, 145, 470, 260, formula, self.code, title="path state")
        d.text((740, 465), "throughput — это сколько энергии еще несет путь.\nradiance — накопленный ответ для пикселя.", font=self.small, fill="#f6efe3")

    def scene_pixel_buffers(self, img: Image.Image, t: float) -> None:
        d = ImageDraw.Draw(img)
        header(d, "11. Где хранится цвет пикселя")
        boxes = [
            ("frame_radiance[y,x,RGB]", "один новый случайный GPU-кадр\n1 sample per pixel", 80, 160, "#2f80ed"),
            ("realtime_accumulation", "сумма всех frame_radiance\nнакопленных в GUI", 450, 160, "#27ae60"),
            ("current_artifact.radiance", "физическая яркость\nсохраняется в HDR", 820, 160, "#f2994a"),
            ("current_artifact.display", "tone mapped 0..255\nсохраняется в PNG/PPM", 450, 450, "#bb6bd9"),
        ]
        for title, body, x, y, color in boxes:
            card(d, x, y, 330, 150, color, "#e6eef9")
            d.text((x + 20, y + 24), title, font=self.small, fill="white")
            d.text((x + 20, y + 65), body, font=self.tiny, fill="#f6efe3")
        arrow(d, (410, 235), (450, 235), "#f8d17a", width=4)
        arrow(d, (780, 235), (820, 235), "#f8d17a", width=4)
        arrow(d, (985, 315), (655, 450), "#f8d17a", width=4)
        d.text((170, 630), "Пиксель на экране = tone_map( accumulation / SPP ).", font=self.text, fill="#f8d17a")

    def scene_spp_accumulation(self, img: Image.Image, t: float) -> None:
        d = ImageDraw.Draw(img)
        header(d, "12. Почему шум уменьшается")
        centers = [(240, 345), (640, 345), (1040, 345)]
        spp = [1, 4, 16]
        for (cx, cy), n in zip(centers, spp):
            img.paste(self.noisy_patch(260, 260, n), (cx - 130, cy - 130))
            d.rectangle((cx - 130, cy - 130, cx + 130, cy + 130), outline="#e6eef9", width=2)
            d.text((cx - 65, cy + 155), f"SPP = {n}", font=self.h2, fill="#f6efe3")
            d.text((cx - 95, cy + 195), f"noise ~ 1/sqrt({n})", font=self.small, fill="#b9d7ff")
        formula = """radiance_avg =
(sample_1 + sample_2 + ... + sample_N) / N"""
        code_box(d, 310, 565, 660, 82, formula, self.code_big, title="")

    def scene_tone_hdr(self, img: Image.Image, t: float) -> None:
        d = ImageDraw.Draw(img)
        header(d, "13. HDR и PNG — это разные данные")
        code_box(d, 80, 145, 520, 300, """HDR:
current_artifact.radiance

значения могут быть больше 1.0
это физическая яркость сцены
нужно для анализа и LumiVue-подобных viewer'ов""", self.code, title="real radiance")
        code_box(d, 680, 145, 520, 300, """PNG / PPM:
current_artifact.display

positive = clip(radiance, 0)
scale = percentile(...)
display = pow(positive / scale, 1/gamma) * 255""", self.code, title="tone mapped image")
        arrow(d, (600, 295), (680, 295), "#f8d17a", width=5)
        d.text((180, 545), "HDR сохраняет свет как числа.", font=self.text, fill="#f8d17a")
        d.text((680, 545), "PNG сохраняет то, что удобно видеть глазами.", font=self.text, fill="#f8d17a")

    def scene_gui_cli(self, img: Image.Image, t: float) -> None:
        d = ImageDraw.Draw(img)
        header(d, "14. Как этим управлять")
        gui = """GUI:
GPU PathTrace старт
  -> бесконечное progressive накопление

GPU финал до SPP
  -> догоняет до render.samples_per_pixel

Сохранить PNG/HDR/PPM
  -> сохраняет текущий накопленный кадр"""
        cli = """CLI:
image-lab4-cli --gpu-pathtrace
  -> берет SPP из JSON

image-lab4-cli --gpu-pathtrace --frames 100
  -> вручную накопить 100 кадров"""
        code_box(d, 90, 140, 520, 390, gui, self.code, title="PySide6 GUI")
        code_box(d, 675, 140, 520, 390, cli, self.code, title="terminal")
        d.text((120, 598), "Камера в GUI вращается стрелками ← / →, accumulation после поворота сбрасывается.", font=self.small, fill="#dbe9ff")

    def scene_code_map(self, img: Image.Image, t: float) -> None:
        d = ImageDraw.Draw(img)
        header(d, "15. Где это находится в коде")
        rows = [
            ("config_loader.py", "читает JSON и создает SceneConfig"),
            ("models/scene.py", "Material, Triangle, PointLight, Scene, RenderArtifact"),
            ("math/geometry.py", "reflect, triangle_area, triangle_normal, sample_point_on_triangle"),
            ("path_tracer.py", "CPU/Torch версия и расчет light_probabilities"),
            ("taichi_progressive_path_tracer.py", "GPU kernel: лучи, пересечения, свет, bounce"),
            ("ui/main_window.py", "GUI, accumulation, сохранение текущего кадра"),
            ("report/exporters.py", "PNG, PPM, HDR export"),
        ]
        y = 125
        for file_name, body in rows:
            card(d, 110, y, 1060, 64, "#111f30", "#2d5378")
            d.text((135, y + 18), file_name, font=self.small, fill="#f8d17a")
            d.text((470, y + 18), body, font=self.small, fill="#e6eef9")
            y += 72

    def scene_final_script(self, img: Image.Image, t: float) -> None:
        d = ImageDraw.Draw(img)
        header(d, "16. Короткий рассказ на сдаче")
        bullets = [
            "Сцена задается JSON: камера, материалы, треугольники, area и point lights.",
            "Path tracer выпускает случайные лучи из камеры через пиксели.",
            "Ближайшее пересечение ищется по треугольникам методом Moller-Trumbore.",
            "Материал выбирает diffuse или mirror bounce, а throughput хранит энергию пути.",
            "Источник выбирается по значимости; point light учитывается через intensity.",
            "Цвет пикселя — среднее многих samples; чем больше SPP, тем меньше шум.",
            "HDR хранит физическую radiance, PNG хранит tone-mapped картинку.",
        ]
        y = 126
        for i, line in enumerate(bullets, start=1):
            d.text((115, y), f"{i}.", font=self.h2, fill="#f8d17a")
            d.text((165, y + 5), line, font=self.text, fill="#f6efe3")
            y += 70
        center_text(d, "Если забыли формулу: свет = вклад источника / вероятность выбора / distance² * cos", self.small, 652, "#b9d7ff")

    def draw_simple_scene(self, d: ImageDraw.ImageDraw, t: float, highlight: str) -> None:
        d.polygon([(130, 560), (760, 560), (940, 280), (340, 280)], fill="#252f37", outline="#566879")
        d.polygon([(420, 430), (585, 430), (585, 275), (420, 275)], fill="#c86a44", outline="#f6efe3", width=2)
        d.text((410, 444), "diffuse cube", font=self.tiny, fill="#f6efe3")
        d.polygon([(565, 430), (700, 395), (690, 240), (555, 275)], fill="#1f7f44", outline="#8dff9f", width=3)
        d.text((575, 446), "green mirror", font=self.tiny, fill="#8dff9f")
        d.ellipse((190, 480, 245, 535), fill="#e4573d", outline="#ffffff", width=2)
        d.text((170, 542), "camera", font=self.tiny, fill="#f6efe3")
        d.rectangle((690, 170, 770, 205), fill="#ffd66b", outline="#fff3ba", width=2)
        d.text((695, 138), "A1", font=self.tiny, fill="#ffd66b")
        d.rectangle((920, 230, 1000, 265), fill="#7db7ff", outline="#cde5ff", width=2)
        d.text((925, 198), "A2", font=self.tiny, fill="#7db7ff")
        d.ellipse((785, 300, 825, 340), fill="#ff9f43", outline="#ffe1b5", width=2)
        d.text((778, 346), "P1", font=self.tiny, fill="#ffb86b")


    def noisy_patch(self, w: int, h: int, spp: int) -> Image.Image:
        import numpy as np

        rng = np.random.default_rng(100 + spp)
        base = np.zeros((h, w, 3), dtype=np.float32)
        yy, xx = np.mgrid[0:h, 0:w]
        base[..., 0] = 0.25 + 0.35 * (xx / w)
        base[..., 1] = 0.35 + 0.30 * (yy / h)
        base[..., 2] = 0.45
        base[(xx - w * 0.55) ** 2 + (yy - h * 0.45) ** 2 < (w * 0.22) ** 2] += [0.35, 0.25, 0.05]
        noise = rng.normal(0.0, 0.32 / math.sqrt(spp), base.shape)
        arr = (np.clip(base + noise, 0, 1) * 255).astype("uint8")
        return Image.fromarray(arr, "RGB")


def build_mp4(frames_dir: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    input_pattern = str(frames_dir / "frame_%05d.png")
    cmd = [
        ffmpeg,
        "-y",
        "-framerate",
        str(FPS),
        "-i",
        input_pattern,
        "-vf",
        "format=yuv420p",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        str(OUT_MP4),
    ]
    subprocess.run(cmd, check=True)


def font(size: int, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_MONO if mono else FONT_BOLD if bold else FONT_REGULAR
    return ImageFont.truetype(path, size=size)


def header(d: ImageDraw.ImageDraw, text: str) -> None:
    d.text((72, 58), text, font=font(38, bold=True), fill="#f6efe3")
    d.line((72, 112, WIDTH - 72, 112), fill="#2d5378", width=2)


def card(d, x, y, w, h, fill, outline) -> None:
    d.rounded_rectangle((x, y, x + w, y + h), 18, fill=fill, outline=outline, width=2)


def code_box(d, x, y, w, h, text, fnt, title="") -> None:
    card(d, x, y, w, h, "#0b111a", "#395a7e")
    if title:
        d.text((x + 20, y + 14), title, font=font(17, bold=True), fill="#f8d17a")
        y_text = y + 46
    else:
        y_text = y + 18
    d.multiline_text((x + 20, y_text), text, font=fnt, fill="#dbe9ff", spacing=7)


def pill(d, xy, text, fnt, fill, outline) -> None:
    x, y = xy
    bbox = d.textbbox((0, 0), text, font=fnt)
    w = bbox[2] - bbox[0] + 32
    h = bbox[3] - bbox[1] + 20
    d.rounded_rectangle((x, y, x + w, y + h), h // 2, fill=fill, outline=outline, width=2)
    d.text((x + 16, y + 8), text, font=fnt, fill="#f6efe3")


def center_text(d, text, fnt, y, fill) -> None:
    bbox = d.textbbox((0, 0), text, font=fnt)
    d.text(((WIDTH - (bbox[2] - bbox[0])) / 2, y), text, font=fnt, fill=fill)


def arrow(d, start, end, color, width=3) -> None:
    d.line((start, end), fill=color, width=width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    size = 16
    p1 = (end[0] - size * math.cos(angle - 0.45), end[1] - size * math.sin(angle - 0.45))
    p2 = (end[0] - size * math.cos(angle + 0.45), end[1] - size * math.sin(angle + 0.45))
    d.polygon([end, p1, p2], fill=color)


def lerp_point(a, b, t):
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def ease_step(t, a, b) -> float:
    if t <= a:
        return 0.0
    if t >= b:
        return 1.0
    x = (t - a) / max(b - a, 1e-9)
    return x * x * (3 - 2 * x)


def blend(a: str, b: str, t: float) -> str:
    ca = tuple(int(a[i : i + 2], 16) for i in (1, 3, 5))
    cb = tuple(int(b[i : i + 2], 16) for i in (1, 3, 5))
    cc = tuple(int(ca[i] + (cb[i] - ca[i]) * t) for i in range(3))
    return "#{:02x}{:02x}{:02x}".format(*cc)


if __name__ == "__main__":
    main()
