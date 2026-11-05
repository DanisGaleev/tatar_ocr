import json
from pathlib import Path

GT_DATA = {
    "blank_01": {
        "student_name": "",
        "questions": {
            "1": "КИТАПӘРМ",
            "2": "ЫЦЭЩЦЯИҺ",
            "3": "ӨОЖҖЛДМҢ",
            "4": "УРМАНЛАР",
        }
    },
    "blank_02": {
        "student_name": "",
        "questions": {
            "1": "АБӘЙКАХЖ",
            "2": "ӨХЧШУЙҢҖ",
            "3": "ӨЛАДЕНӘТ",
            "4": "ХӘУЫРӘКИ",
        }
    },
    "blank_03": {
        "student_name": "",
        "questions": {
            "1": "КӨЛМИӨФИ",
            "2": "КАРАМЫЙӨ",
            "3": "УРАМНАРО",
            "4": "АЛМЫЙЫОЛ",
        }
    },
    "blank_04": {
        "student_name": "",
        "questions": {
            "1": "ВАЮГЙМКЮ",
            "2": "БАСФЯРЧҺ",
            "3": "ЭТВПӨЩХШ",
            "4": "ҺТҮУҢҖЦЖ",
        }
    },
    "blank_05": {
        "student_name": "",
        "questions": {
            "1": "ЮЫЪЬХМНК",
            "2": "ГЛАӨЩДХЫ",
            "3": "ӨЩХХЁЕҖХ",
            "4": "БӨХЁҖХЕЁ",
        }
    },
    "blank_06": {
        "student_name": "",
        "questions": {
            "1": "КИТАПКАД",
            "2": "ӨСТӘЛДҢҖ",
            "3": "БЫЛОГДӨЕ",
            "4": "ЙЮЯВШЦИҺ",
        }
    },
    "blank_07": {
        "student_name": "",
        "questions": {
            "1": "АЛМЫЙДҖХ",
            "2": "КӨЛМИҢМЫ",
            "3": "КАРАМЫЙК",
            "4": "УРАМНАРҮ",
        }
    },
    "blank_08": {
        "student_name": "",
        "questions": {
            "1": "ЩФВЩДГЮБ",
            "2": "НКРАЕЁПЭ",
            "3": "ЫҢКОӨҖХҢ",
            "4": "ҮУРҖАЛГД",
        }
    },
    "blank_09": {
        "student_name": "",
        "questions": {
            "1": "ӨСЮНЕИТА",
            "2": "КУҮЮНПБП",
            "3": "ҮАНМЮЯЪЫ",
            "4": "РЕЫЬЯАЗҮ",
        }
    },
    "blank_10": {
        "student_name": "",
        "questions": {
            "1": "КИТАПКА",
            "2": "ӨСТӘЛДӘН",
            "3": "БАЛАЛАР",
            "4": "УРМАННАР",
            "5": "ДӨРЕС",
            "6": "РӘХМӘТ",
            "7": "УКЫТУЧЫНЫҢ",
        }
    },
    "blank_11": {
        "student_name": "",
        "questions": {
            "1": "АӘБВГДЕЁ",
            "2": "ЖЗИКЛМНО",
            "3": "ПРСТУҮФХ",
            "4": "ЦЧШЩЪЫЬЭ",
            "5": "ЮЯӨҺЙҢ",
            "6": "ҖКӘӨПРС",
            "7": "МНАГЁВҮУТФ",
        }
    },
    "blank_12": {
        "student_name": "",
        "questions": {
            "1": "ЪЗЛБЭДХЮ",
            "2": "УҮШРСПТФ",
            "3": "ӘҢҺҖӨҮҮХ",
            "4": "ВГЬЯҖҢӨТ",
            "5": "МНОПСБ",
            "6": "ОЙЕИКМН",
            "7": "ЖЦФҮОАКВПС",
        }
    },
}

def build_entry(image_id: str, data: dict) -> dict:
    name_str = data["student_name"]
    name_cells = []
    for idx in range(15):
        char = name_str[idx] if idx < len(name_str) else ""
        name_cells.append({
            "cell_idx": idx + 1,
            "char": char,
            "is_empty": (char == "" or char == " ")
        })

    questions_list = []
    questions_dict = {}
    for q_str, text in data["questions"].items():
        q_num = int(q_str)
        cells = []
        for c_idx, ch in enumerate(text):
            cells.append({
                "cell_idx": c_idx + 1,
                "char": ch,
                "is_empty": False
            })
        questions_list.append({
            "q_num": q_num,
            "text": text,
            "cells": cells
        })
        questions_dict[q_str] = text

    return {
        "image_id": image_id,
        "pipeline": "ground_truth",
        "student_name": {
            "text": name_str,
            "cells": name_cells
        },
        "questions": questions_list,
        "questions_dict": questions_dict
    }

def main():
    gt_dir = Path(__file__).resolve().parent / "ground_truth"
    gt_dir.mkdir(parents=True, exist_ok=True)

    all_entries = {}
    for img_id, raw_info in GT_DATA.items():
        entry = build_entry(img_id, raw_info)
        all_entries[img_id] = entry
        file_path = gt_dir / f"{img_id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(entry, f, ensure_ascii=False, indent=2)
        print(f"Wrote {file_path}")

    all_path = gt_dir / "ground_truth_all.json"
    with open(all_path, "w", encoding="utf-8") as f:
        json.dump(all_entries, f, ensure_ascii=False, indent=2)
    print(f"Wrote {all_path}")

    test_path = gt_dir / "ground_truth_test.json"
    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(all_entries, f, ensure_ascii=False, indent=2)
    print(f"Wrote {test_path}")

if __name__ == "__main__":
    main()
