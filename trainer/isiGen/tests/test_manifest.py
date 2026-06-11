from src.core.manifest import Manifest, ManifestRecord, MaskPrompt


def _rec(i="abc123def456"):
    return ManifestRecord(id=i, sha256=i * 5 + "0" * 4, image=f"raw/p/{i}.jpg",
                          class_name="palette", width=800, height=600)


def test_roundtrip_and_atomic_save(tmp_path):
    m = Manifest(tmp_path)
    m.upsert(_rec())
    m.save()
    assert (tmp_path / "manifest.jsonl").exists()
    assert not (tmp_path / "manifest.jsonl.tmp").exists()    # atomic replace
    m2 = Manifest.load(tmp_path)
    assert m2.get("abc123def456").class_name == "palette"


def test_unknown_keys_tolerated(tmp_path):
    m = Manifest(tmp_path)
    m.upsert(_rec())
    m.save()
    raw = (tmp_path / "manifest.jsonl").read_text().strip()
    (tmp_path / "manifest.jsonl").write_text(
        raw[:-1] + ', "future_field": [1, 2, 3]}\n')
    m2 = Manifest.load(tmp_path)
    assert m2.get("abc123def456") is not None                # forward-compatible


def test_mask_prompt_roundtrip(tmp_path):
    r = _rec()
    r.mask_prompts = [MaskPrompt(kind="point", class_name="palette", xy=[10.5, 20.0]),
                      MaskPrompt(kind="box", class_name="carton", xyxy=[1, 2, 3, 4])]
    m = Manifest(tmp_path)
    m.upsert(r)
    m.save()
    got = Manifest.load(tmp_path).get(r.id)
    assert got.mask_prompts[0].kind == "point" and got.mask_prompts[0].xy == [10.5, 20.0]
    assert got.mask_prompts[1].xyxy == [1, 2, 3, 4]


def test_active_skips_excluded(tmp_path):
    m = Manifest(tmp_path)
    a, b = _rec("a" * 12), _rec("b" * 12)
    b.excluded = True
    m.upsert(a)
    m.upsert(b)
    assert [r.id for r in m.active()] == ["a" * 12]
