import json
from datetime import datetime

def generate_items(count, start_id=1):
    items = []

    for i in range(count):
        item_id = start_id + i
        now = datetime.now().strftime("%H:%M:%S %d.%m.%Y")

        item = {
            "id": item_id,
            "name": str(item_id % 10),
            "coordinates": {
                "x": float(item_id % 10),
                "y": item_id % 10
            },
            "creationDate": now,
            "health": float(item_id % 10),
            "height": item_id % 10,
            "category": "DREADNOUGHT" if item_id % 2 == 0 else "APOTHECARY",
            "meleeWeapon": "POWER_SWORD" if item_id % 2 == 0 else "POWER_FIST",
        }

        # иногда добавляем chapter
        if item_id % 3 == 0:
            item["chapter"] = {
                "name": str(item_id % 10),
                "parentLegion": str(item_id % 5)
            }

        items.append(item)

    return items


# настройки
COUNT = 1000        # сколько элементов
START_ID = 10       # с какого id начать

data = generate_items(COUNT, START_ID)

with open("output.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print("готово")