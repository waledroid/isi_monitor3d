from src.core.manifest import ManifestRecord
from src.core.project import ProjectConfig
from src.stages.captioning.template import TemplateCaptioner


def _project():
    return ProjectConfig(name="p", classes=[
        {"name": "palette", "trigger": "ISI_PLT", "color": [1, 2, 3]}])


def _rec(i):
    return ManifestRecord(id=i, sha256="0" * 64, image="x.jpg",
                          class_name="palette", width=10, height=10)


def test_caption_contains_trigger_phrase_background():
    cap = TemplateCaptioner(
        patterns=["a photo of {trigger} {class_phrase}, {background}"],
        class_phrases={"palette": "wooden transport pallet"},
        backgrounds=["a busy warehouse aisle"])
    text = cap.caption(_rec("aaaaaaaaaaaa"), _project())
    assert "ISI_PLT" in text and "wooden transport pallet" in text
    assert "a busy warehouse aisle" in text


def test_deterministic_per_id_and_varied_across_ids():
    cap = TemplateCaptioner(backgrounds=[f"background {i}" for i in range(50)])
    p = _project()
    one = cap.caption(_rec("aaaaaaaaaaaa"), p)
    assert one == cap.caption(_rec("aaaaaaaaaaaa"), p)       # stable re-runs
    texts = {cap.caption(_rec(f"{i:012d}"), p) for i in range(20)}
    assert len(texts) > 5                                     # backgrounds vary
