# Steam Game Recommendation Coursework

В работе сравниваются классические рекомендательные бейзлайны, prompt-only LLM бейзлайны, CatBoost с семантическими признаками, QLoRA fine-tuning и ReLLa-lite варианты. В репозитории оставлены код, ноутбуки, финальная PDF-записка и небольшие файлы с метриками и summary. Большие исходные данные, обработанные датасеты, prediction-файлы, checkpoint-и, adapter weights и локальные кэши намеренно не включены.

## Структура репозитория

- `scripts/data_preparation/` - скрипты аудита данных, объединения метаданных, построения temporal split-ов и instruction-датасетов.
- `scripts/baselines/` - классические бейзлайны: majority, popularity, content similarity, logistic regression, CatBoost и ItemKNN.
- `scripts/llm_baselines/` - prompt-only zero-shot и few-shot LLM бейзлайны.
- `notebooks/semantic_features/` - генерация semantic features и эксперименты CatBoost с семантическими признаками.
- `notebooks/qlora/` - ноутбуки обучения и оценки обычного QLoRA-подхода.
- `notebooks/rella_lite/` - ReLLa-lite dataset, continuation training и evaluation.
- `notebooks/analysis/` - long-tail и cold-start group analysis.
- `outputs/metrics/` - небольшие итоговые таблицы с метриками.
- `outputs/summaries/` - небольшие summary-файлы и краткие отчеты по экспериментам.
- `reports/final/` - финальная PDF-записка.
- `docs/` - описание нужных локальных данных и индекс артефактов.

## Установка зависимостей

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Скрипты и ноутбуки ожидают, что большие локальные данные лежат в путях, описанных в `docs/data_files.md`.

## Финальная записка

Финальная PDF-записка лежит здесь:

```text
reports/final/coursework_final_note.pdf
```

## Итоговые метрики

Итоговые таблицы с метриками собраны здесь:

```text
outputs/metrics/
```

Краткий индекс метрик, summary-файлов и исключенных больших артефактов находится в `docs/artifact_index.md`.
