import pytest
from src.core.project import TEMPLATE_PATH, ClassSpec, ProjectConfig, create_project, load_project


def test_template_is_valid():
    import yaml
    data = yaml.safe_load(TEMPLATE_PATH.read_text())
    cfg = ProjectConfig.model_validate(data)
    assert len(cfg.classes) >= 1
    assert "generation" in cfg.phases       # reserved phase keys present


@pytest.mark.parametrize("field,vals", [
    ("name", ["dup", "dup"]),
    ("trigger", ["T1", "T1"]),
])
def test_duplicate_class_fields_rejected(field, vals):
    classes = [{"name": f"c{i}", "trigger": f"T{i}", "color": [i, i, i]} for i in range(2)]
    for i, v in enumerate(vals):
        classes[i][field] = v
    with pytest.raises(ValueError, match="must be unique"):
        ProjectConfig(name="x", classes=classes)


def test_duplicate_color_rejected():
    with pytest.raises(ValueError, match="must be unique"):
        ProjectConfig(name="x", classes=[
            {"name": "a", "trigger": "TA", "color": [1, 2, 3]},
            {"name": "b", "trigger": "TB", "color": [1, 2, 3]}])


def test_create_and_reload(tmp_path):
    pdir = create_project(tmp_path, "demo",
                          [ClassSpec(name="palette", trigger="ISI_PLT", color=[220, 40, 40])])
    cfg = load_project(pdir)
    assert cfg.name == "demo" and cfg.class_names() == ["palette"]
    assert (pdir / "raw").is_dir() and (pdir / "maps" / "mask").is_dir()
    with pytest.raises(FileExistsError):
        create_project(tmp_path, "demo", [ClassSpec(name="x", trigger="T", color=[1, 1, 1])])


def test_unknown_class_lookup_raises(tmp_path):
    cfg = ProjectConfig(name="x", classes=[{"name": "a", "trigger": "T", "color": [1, 2, 3]}])
    with pytest.raises(KeyError, match="unknown class"):
        cfg.class_by_name("nope")
