# Project map
- Standalone Go HTTP microservice for UK/European licence-plate detection and OCR.
- Public contract: health endpoint plus multipart image extraction endpoint; inference stays local via ONNX Runtime.
- Pipeline invariants: decode image -> YOLOv9 plate detection -> select highest-confidence valid box -> padded crop -> grayscale MobileViT OCR -> validate text -> rank accepted results across images by detector confidence.
- Exact model and preprocessing invariants belong with implementation tests and form part of the service's compatibility contract.
- Read `mem:tech_stack` for native/runtime pins, `mem:conventions` before changes, `mem:suggested_commands` for workflows, and `mem:task_completion` before handoff.
