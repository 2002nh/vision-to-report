# Vision-to-Report: YOLO + Transformer Inspection Notes

**A small experiment in connecting object detection with language models.**

Most object detection demos end with bounding boxes on an image. I wanted to try the next step: taking those predictions and turning them into a short, readable inspection note. This project is my starting point for exploring how computer vision and Transformers can work together in a simple workflow.

The idea is straightforward: **detections in → structured summary → optional Transformer rewrite**. It's not a production inspection system, but it's useful for learning about model integration, confidence thresholds, and the limits of generated text.

## What it does

- Reads object detection results from JSON or exports results from a YOLO model.
- Filters low-confidence predictions and counts the remaining detected classes.
- Produces an inspection note with the original counts and confidence information.
- Optionally uses a Hugging Face Transformer (`google/flan-t5-small`) to rewrite those findings in natural language.
- Keeps the numerical summary separate from generated text so it's easier to notice mistakes.

**Tech:** Python · YOLO (Ultralytics) · Hugging Face Transformers · PyTorch · JSON

## Project flow

```text
Image ── YOLO model (optional) ── detection JSON
                                     │
                      confidence filtering + grouping
                                     │
                           deterministic report
                                     │
                      optional FLAN-T5 rewrite
```

The repository includes **hand-written example detections**, so you can try it without downloading a model. The example is fictional; it is *not* evidence of defect-detection accuracy.

## Quick start (no ML model needed)

Use Python 3.10 or newer:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -e .
python -m inspection_report.cli --input examples/sample_detections.json
```

To save the report:

```bash
python -m inspection_report.cli --input examples/sample_detections.json --output outputs/report.txt
```

### Try the Hugging Face Transformer

```bash
python -m pip install -e ".[llm]"
python -m inspection_report.cli --input examples/sample_detections.json --rewrite
```

The first run downloads Google's **FLAN-T5-small** model (approximately 300 MB of model weights). The generated sentence is experimental. It may omit or invent details, so compare it with the counts in the original report. A small language model may also produce awkward wording; that is part of the experiment.

### Use your own YOLO detections

If you already have a trained YOLO model, you can export its predictions:

```bash
python -m pip install -e ".[yolo]"
python scripts/run_yolo.py --image path/to/image.jpg --weights path/to/best.pt --output outputs/detections.json
python -m inspection_report.cli --input outputs/detections.json --rewrite
```

> **Important:** Pretrained YOLO object detection weights for everyday objects **do not automatically recognize industrial scratches, corrosion, or missing fasteners**. Real defect detection requires appropriate labeled data, trained weights, and evaluation. No custom detector or training dataset is bundled here.

## Example of the basic report

This output comes from the fictional example JSON and the deterministic report generator:

```text
VISUAL INSPECTION NOTES — DEMO
Image: demo_machine_part_001.jpg
Detection source: hand-written sample data (not actual YOLO inference)
Confidence threshold: 0.40
Detections used: 4 of 5

Findings:
- corrosion: 1 detection(s); mean confidence 0.83
- missing bolt: 1 detection(s); mean confidence 0.63
- surface scratch: 2 detection(s); mean confidence 0.86
```

## Why keep a deterministic report?

A text model can make a description more natural, but it can also hallucinate or drop information. I didn't want the LLM to be the only source of truth. The code calculates counts directly from the detection data and labels the Transformer rewrite as experimental.

## Files

```text
vision-to-report/
├── examples/sample_detections.json      # fictional detections for testing
├── scripts/run_yolo.py                 # optional image → YOLO JSON conversion
├── src/inspection_report/
│   ├── core.py                         # validate, filter, summarize
│   ├── llm.py                          # optional FLAN-T5 rewriting
│   └── cli.py                          # command line interface
├── tests/test_core.py                  # offline unit tests
├── pyproject.toml
└── README.md
```

## Tests

```bash
python -m unittest discover -s tests -v
```

These cover validation, confidence filtering, empty results, and prompt construction. They don't claim anything about model accuracy.

## What I'd like to improve

- Test the pipeline with a real labeled industrial image dataset and fine-tuned YOLO weights.
- Evaluate detection performance (precision, recall, mAP) instead of relying on example data.
- Compare generated inspection notes against human-reviewed references.
- Add bounding-box previews and eventually a simple UI.

## Limitations

This is an educational prototype. Example detections are fictional, the Transformer model is pretrained rather than fine-tuned here, and no real industrial accuracy or safety performance has been measured. A person should always check both the detections and the original images.

**Model references:** [Ultralytics YOLO](https://docs.ultralytics.com/modes/predict/) · [Google FLAN-T5-small](https://huggingface.co/google/flan-t5-small)
