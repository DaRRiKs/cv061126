import random
from collections import Counter, defaultdict

def generate_questions(train_annotations, val_annotations, count=100, seed=42):
    categories = {c["id"]:c["name"] for c in train_annotations["categories"]}
    train_objects, val_objects = defaultdict(set), defaultdict(set)
    for ann in train_annotations["annotations"]:
        train_objects[ann["image_id"]].add(ann["category_id"])
    for ann in val_annotations["annotations"]:
        val_objects[ann["image_id"]].add(ann["category_id"])
    frequency, cooccurrence = Counter(), Counter()
    for present in train_objects.values():
        frequency.update(present)
        for first in present:
            for second in present:
                if first != second:
                    cooccurrence[first,second] += 1
    eligible = sorted(i for i, present in val_objects.items() if len(present)>=3 and len(categories)-len(present)>=3)
    rng = random.Random(seed)
    selected = rng.sample(eligible,count)
    rows = []
    for image_id in selected:
        present = val_objects[image_id]
        positive = rng.sample(sorted(present),3)
        absent = sorted(set(categories)-present)
        negatives = {
            "random":rng.sample(absent,3),
            "popular":sorted(absent,key=lambda c:(-frequency[c],c))[:3],
            "adversarial":sorted(absent,key=lambda c:(-sum(cooccurrence[p,c]/max(frequency[p],1) for p in present),c))[:3]
        }
        for mode, negative in negatives.items():
            assert len(set(negative))==3 and not (set(negative)&present)
            for label, objects in [(1,positive),(0,negative)]:
                for category_id in objects:
                    name = categories[category_id]
                    rows.append({"question_id":f"{image_id}_{mode}_{category_id}","image_id":image_id,
                                 "split":"val","mode":mode,"category_id":category_id,"object":name,
                                 "label":label,"question":f"Is there a {name} in the image?"})
    statistics = {"source":"COCO train2017 image-level object categories",
                  "train_images":len(train_objects),"seed":seed,"selected_images":selected,
                  "frequency":{str(k):v for k,v in frequency.items()},
                  "cooccurrence":{f"{a},{b}":v for (a,b),v in cooccurrence.items()}}
    return rows, statistics

if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path
    parser = argparse.ArgumentParser()
    parser.add_argument("train_annotations", type=Path)
    parser.add_argument("val_annotations", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows, statistics = generate_questions(json.loads(args.train_annotations.read_text()), json.loads(args.val_annotations.read_text()))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "questions.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "object_statistics.json").write_text(json.dumps(statistics, indent=2), encoding="utf-8")
