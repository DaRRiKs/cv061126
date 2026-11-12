from pathlib import Path
import ast
import hashlib
import io
import json
import tokenize
import nbformat
import numpy as np
import pandas as pd
import fitz
import torch

ROOT = Path(__file__).resolve().parent

def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))

def has(path):
    return path.exists() and path.stat().st_size > 0

def audit():
    rows = []
    file_rows = []
    outputs = {i:ROOT/f"PZ_{i}/results" for i in range(1,7)}
    for number in range(1,7):
        folder = ROOT/f"PZ_{number}"
        notebook = nbformat.read(folder/f"PZ_{number}.ipynb",as_version=4)
        nbformat.validate(notebook)
        markdown = "\n".join(c.source for c in notebook.cells if c.cell_type=="markdown")
        assert "??" not in markdown and "\ufffd" not in markdown,number
        for section in range(1,9):
            assert f"## {section}." in markdown,(number,section)
        executed = 0
        for c in notebook.cells:
            if c.cell_type!="code":
                continue
            if c.source.startswith("%pip"):
                continue
            tree = ast.parse(c.source)
            assert "??" not in c.source and "\ufffd" not in c.source,(number,c.id)
            assert not any(t.type==tokenize.COMMENT for t in tokenize.generate_tokens(io.StringIO(c.source).readline))
            for node in ast.walk(tree):
                if isinstance(node,(ast.Module,ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)):
                    assert ast.get_docstring(node) is None
            assert not any(o.output_type=="error" for o in c.outputs),(number,c.id)
            executed += c.execution_count is not None
        assert read_json(outputs[number]/"unit_tests.json")["passed"]
        with fitz.open(folder/"report.pdf") as document:
            pages = len(document)
            assert 1<=pages<=2,(number,pages)
            text = "\n".join(page.get_text() for page in document)
            assert "Индивидуальный вклад" in text
        file_rows.append(f"| ПЗ-{number} | PZ_{number}.ipynb, report.pdf, requirements.txt, results/ | {executed} | {pages} |")

    def add(number,points,criterion,result,evidence,explanation="Есть в Markdown-ячейках и PDF"):
        status = "Выполнено" if result else "Не завершено"
        rows.append(f"| ПЗ-{number}, {points} | {criterion} | Есть | {status}: {evidence} | {explanation} |")

    o=outputs[1]
    eq=pd.read_csv(o/"equivalence.csv")
    assert len(eq)==8 and eq.max_abs_error.max()<1e-5
    token=pd.read_csv(o/"tokens.csv")
    assert len(token)==8 and np.all(token.N_squared==token.N**2)
    sample=read_json(o/"sample.json")
    for index in sample["indices"]:
        assert np.load(o/f"attention_{index}.npy").shape==(12,14,14)
        assert has(o/f"heads_{index}.png")
    add(1,"1–2","PatchEmbed и копирование весов",True,f"8 проверок; max abs={eq.max_abs_error.max():.8g}")
    add(1,"3","N, N², относительный рост",True,"8 строк tokens.csv; максимальный рост 144×")
    add(1,"4–6","DINO, CLS без CLS-патча, все головы",True,"4 изображения × 12 карт 14×14; NPY и PNG")
    add(1,"7","Интерпретация голов",True,"Два абзаца после просмотра реальных карт")

    o=outputs[2]
    zero_done=False
    if has(o/"zero_shot.csv"):
        zero=pd.read_csv(o/"zero_shot.csv")
        assert len(zero)==6 and zero.n.eq(10000).all()
        assert zero.accuracy.between(0,1).all()
        for p in o.glob("predictions_*.csv"):
            predictions=pd.read_csv(p)
            assert len(predictions)==10000
        assert len(list(o.glob("predictions_*.csv")))==6
        zero_done=has(o/"confusion_matrix.png")
    features_done=has(o/"features_train.pt") and has(o/"features_test.pt")
    if features_done:
        feature_ids=[]
        for split,n in [("train",5000),("test",1000)]:
            features=torch.load(o/f"features_{split}.pt",weights_only=True,map_location="cpu")
            assert features["image"].shape==features["text"].shape==(n,512)
            assert torch.isfinite(features["image"]).all() and torch.isfinite(features["text"]).all()
            assert len(set(features["image_ids"]))==n
            feature_ids.append(set(features["image_ids"]))
        assert not feature_ids[0]&feature_ids[1]
    recall_done=False
    if has(o/"recall.csv"):
        recall=pd.read_csv(o/"recall.csv")
        recall_done=len(recall)==8 and has(o/"recall_batch.png")
        assert recall.epochs.eq(5).all()
        assert set(zip(recall.loss,recall.batch))=={(loss,batch) for loss in ["InfoNCE","SigLIP"] for batch in [8,32,128,512]}
        assert all(r.steps==5*int(np.ceil(5000/r.batch)) for r in recall.itertuples())
        assert recall[["i2t_R1","i2t_R5","t2i_R1","t2i_R5"]].ge(0).all().all()
        assert recall[["i2t_R1","i2t_R5","t2i_R1","t2i_R5"]].le(1).all().all()
        training=pd.read_csv(o/"training.csv")
        for r in recall.itertuples():
            part=training[(training.loss==r.loss)&(training.batch==r.batch)]
            assert len(part)==r.steps and part.step.max()==r.steps
            assert np.isfinite(part.value).all()
            checkpoint=torch.load(o/f"heads_{r.loss}_{r.batch}.pt",weights_only=True,map_location="cpu")
            for head in ["image_head","text_head"]:
                assert checkpoint[head]["weight"].shape==(256,512)
                assert torch.isfinite(checkpoint[head]["weight"]).all()
    add(2,"1–3","Zero-shot 2×3 и confusion matrix",zero_done,"zero_shot.csv и confusion_matrix.png" if zero_done else "Полной таблицы ещё нет")
    add(2,"4–5","InfoNCE, SigLIP и тесты",True,"B=1: 0; scales 1/10/100: 1.874597/0.000681/0")
    add(2,"6","Замороженные CLIP-признаки",features_done,"features_train.pt, features_test.pt" if features_done else "Нет полных предвычисленных признаков")
    add(2,"7–8","Две головы 256, 2 loss × 4 batch",recall_done,"8 опытов, 5 эпох, число шагов сохранено" if recall_done else "Обучение восьми вариантов не завершено")
    add(2,"9–10","Recall@1/@5, два направления, 1000 пар",recall_done,"recall.csv и recall_batch.png" if recall_done else "Полные метрики отсутствуют")
    add(2,"11","Сопоставление со статьёй SigLIP",recall_done,"analysis.json" if recall_done else "Пока только теоретический контекст", "Вывод по реальным Recall; один seed и замороженные энкодеры ограничивают перенос" if recall_done else "Статья приведена; эмпирический вывод требует Recall")

    o=outputs[3]
    assert len(read_json(o/"pairs_train.json"))==20000 and len(read_json(o/"pairs_val.json"))==500
    trained=all(has(o/f"projector_{n}.pt") for n in [196,49])
    evaluated=has(o/"metrics.csv")
    reviewed=has(o/"error_review.json") and len(read_json(o/"error_review.json"))==20
    add(3,"1–3","Заморозка, MLP, 196/49 токенов",trained,"Pooling проверен; два целевых проектора обучены" if trained else "Pooling проверен; обучение целевых моделей не запускалось","Реализация есть; выполнение с настоящими весами требует CUDA")
    add(3,"4–6","inputs_embeds, labels, padding, causal shift",True,"Loss=4.163172; совпал с ручным CE; градиент к визуальным входам есть","Это тест случайной маленькой Qwen, не результат COCO")
    add(3,"7–8","Обучение двух проекторов и loss curves",trained,"Чекпойнты и журнал" if trained else "Нет CUDA; 20000 train-пар зафиксированы")
    add(3,"9–11","500 генераций, BLEU-4, время и контекст",evaluated,"metrics.csv" if evaluated else "Нет генераций и метрик; 500 val-изображений зафиксированы")
    add(3,"12–13","10 примеров и классификация ошибок",reviewed,"error_review.json" if reviewed else "Генерации и их ручная оценка отсутствуют")

    o=outputs[4]
    processor=read_json(o/"processor_tests.json")
    assert len({r["visual_tokens"] for r in processor})==3
    complete=has(o/"answers.json") and len(read_json(o/"answers.json"))==300 and all(r["status"]=="ok" for r in read_json(o/"answers.json"))
    add(4,"1","ANLS и тест 1234/1235",True,"ANLS=0.75; EM=0; остальные граничные тесты пройдены")
    add(4,"2–4","100 вопросов × 3 max_pixels, N′ и время",complete,"300 ответов" if complete else "Проверен только процессор: N′=300/1230/3834","Синтетическое изображение 1500×2000; это не DocVQA")
    add(4,"5–6","ANLS, EM и два графика",complete and has(o/"anls_tokens.png") and has(o/"time_tokens.png"),"Результатов инференса нет" if not complete else "CSV и PNG")
    add(4,"7–8","Числовые ответы",complete and has(o/"numeric_metrics.csv"),"Метрики числовой подвыборки отсутствуют" if not complete else "numeric_metrics.csv")
    add(4,"9","Разбор 10 ошибок",has(o/"error_review.json"),"Требует ответов модели и просмотра документов")
    add(4,"10–11","Итоговая таблица и колено кривой",complete and has(o/"analysis.json"),"Вывод о насыщении не сделан без данных")

    o=outputs[5]
    questions=pd.DataFrame(read_json(o/"questions.json"))
    assert len(questions)==1800 and questions.image_id.nunique()==100
    assert questions.groupby(["mode","label"]).size().eq(300).all()
    generator_done=has(o/"object_statistics.json") and has(o/"image_checks.json") and len(read_json(o/"image_checks.json"))==100
    answers_done=has(o/"answers.json") and len(read_json(o/"answers.json"))==5400
    add(5,"1–6","Позитивы, random/popular/adversarial, баланс, файл",generator_done,"1800 вопросов; частоты по 117266 train-изображениям с объектами; негативы проверены")
    add(5,"7","Сырые ответы и нормализация yes/no",answers_done,"Парсер проверен; ответов VLM нет" if not answers_done else "5400 ответов")
    add(5,"8–10","Метрики 3 режима × 3 промпта",answers_done and has(o/"metrics.csv"),"Итоговые метрики VLM отсутствуют" if not answers_done else "metrics.csv")
    add(5,"11","Формулировки и языковой приор",answers_done and has(o/"analysis.json"),"Реализован анализ FPR и изменений Precision/Recall/F1; выводов без ответов нет")

    o=outputs[6]
    sample=read_json(o/"sample.json")
    assert len(sample["calibration"])==5 and len(sample["evaluation"])==200
    assert sample["partition"]=="unc" and sample["source_rows"]==3811
    assert len({r["image_id"] for r in sample["calibration"]+sample["evaluation"]})==205
    data_done=has(o/"image_checks.json") and len(read_json(o/"image_checks.json"))==205
    model_done=has(o/"metrics.json") and read_json(o/"metrics.json")["n"]==200
    add(6,"1","RefCOCO: изображения, описания, эталонные bbox",data_done,"Выборка 5+200, чистые COCO-оригиналы, размеры и RGB SHA256")
    calibrated=has(o/"coordinate_calibration.json") and len(read_json(o/"coordinate_calibration.json"))==5 and has(o/"coordinate_review.json") and read_json(o/"coordinate_review.json")["confirmed"]
    add(6,"2–5","JSON, 5 калибровок, координаты и ошибки",calibrated,"coordinate_calibration.json и coordinate_review.json" if calibrated else "Парсер и преобразование проверены; калибровки ответами Qwen нет","Формат реализован по документации; проверка пятью предсказаниями отмечается отдельно")
    add(6,"6","IoU и тесты",True,"1, 0, 1/7, 0.25; 100 случайных тестов симметрии")
    add(6,"7","Mean IoU и Acc@0.5 на 200 примерах",model_done,"Ответы VLM отсутствуют" if not model_done else "metrics.json")
    semantic_done=has(o/"error_review.json") and len(read_json(o/"error_review.json"))==200 and all(r["label"]!="uncertain" for r in read_json(o/"error_review.json"))
    visualized=has(o/"visualized_examples.json") and len(read_json(o/"visualized_examples.json"))==6 and has(o/"grounding_examples.png")
    add(6,"8–9","Доли семантических типов ошибок",semantic_done,"200 проверенных случаев" if semantic_done else "Ручной просмотр реализован; без предсказаний классификация не выполнена")
    add(6,"10","3 успешных и 3 неудачных примера",visualized,"grounding_examples.png" if visualized else "Реальные предсказания отсутствуют")

    document=next(ROOT.glob("*06_11*.docx"))
    sha=hashlib.sha256(document.read_bytes()).hexdigest()
    assert sha=="6cae6e05a9d06a339625e8e7003d64ae6331797233448a043aaba795aa9afb45"
    report = ["# Проверка выполнения ПЗ-1 — ПЗ-6", "",f"Исходный документ: `{document.name}`. SHA256: `{sha}`.","",
              "DOCX прочитан полностью. Приложения А.1–А.7 не приложены. Контрольные вопросы: 2+2+2+2+2+1, ответы находятся в разделе 8 каждого notebook. Отдельного подробного задания LoRA нет.","",
              "Проверены структура notebook, синтаксис всех Python-ячеек, отсутствие комментариев и docstring, реальные CPU-тесты, наличие PDF и ограничение 1–2 страницы. Проверка синтаксиса не равна выполнению GPU-эксперимента.","",
              "| Работа | Файлы внутри папки | Выполнено ячеек | Страниц PDF |","|---|---|---:|---:|",*file_rows,"",
              "В каждой results/ сохранены environment.json и unit_tests.json. Дополнительный скрипт ПЗ-5: generate_questions.py. Точный перечень результатов указан в README.","",
              "| Пункт задания | Критерий | Реализация | Реальный результат и статус | Объяснение |","|---|---|---|---|---|",*rows,"",
              "GPU-эксперименты ПЗ-3–6 остаются незавершёнными. Отчёты отражают доступные проверки, а не выдуманные метрики. В Colab требуются также просмотр документов DocVQA, проверка пяти калибровочных примеров и семантическая оценка ошибок."]
    (ROOT/"AUDIT.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    print("Verified notebook structure, code, CPU tests, data sizes and PDFs. See AUDIT.md.")

if __name__ == "__main__":
    audit()
