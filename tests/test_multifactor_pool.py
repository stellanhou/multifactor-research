import hashlib
import json
from pathlib import Path

from crypto_quant.features.factor_expressions import compile_expression
from crypto_quant.research.strategy_research.multifactor_pool import scan_idea_pool


def _card(
    path,
    *,
    card_id,
    expression="perp_close",
    direction=1,
    horizon=24,
    passed_horizons=None,
    source_type="factor_mining",
):
    compiled = compile_expression(expression)
    card = {
        "id": card_id,
        "title": card_id,
        "source_type": source_type,
        "status": "research_idea",
        "source": {"run_id": "pool-test", "candidate_id": card_id},
        "original_claim": {
            "direction": direction,
            "formula": {
                "expression": expression,
                "expanded_expression": compiled.expanded_expression,
                "fields": list(compiled.fields),
                "lookback_hours": compiled.lookback_hours,
            },
        },
        "market_and_horizon": {
            "venue": "Binance",
            "market": "USD-M perpetual",
            "inputs": "1h",
            "passed_horizons": list(passed_horizons if passed_horizons is not None else [horizon]),
        },
        "b_validation_status": "passed",
        "admission_evidence": {"eligible_for_idea_pool": True},
    }
    path.write_text(json.dumps(card), encoding="utf-8")
    return card


def test_scan_preserves_rejections_and_malformed_json_in_sorted_file_order(tmp_path):
    _card(tmp_path / "z-valid.json", card_id="fmv6-valid")
    _card(tmp_path / "b-wrong-horizon.json", card_id="fmv6-horizon", horizon=4)
    _card(
        tmp_path / "c-archive.json",
        card_id="community-card",
        source_type="community",
    )
    (tmp_path / "a-broken.json").write_text("{broken", encoding="utf-8")
    (tmp_path / "d-contract.json").write_text(
        json.dumps({"source_type": "factor_mining", "status": "research_idea"}),
        encoding="utf-8",
    )

    result = scan_idea_pool(tmp_path, 24, 12, {"perp_close"})

    assert [entry["path"].rsplit("/", 1)[-1] for entry in result["entries"]] == [
        "a-broken.json",
        "b-wrong-horizon.json",
        "c-archive.json",
        "d-contract.json",
        "z-valid.json",
    ]
    statuses = {entry["path"].rsplit("/", 1)[-1]: entry["status"] for entry in result["entries"]}
    assert statuses == {
        "a-broken.json": "error",
        "b-wrong-horizon.json": "rejected",
        "c-archive.json": "rejected",
        "d-contract.json": "rejected",
        "z-valid.json": "admitted",
    }
    assert "JSON" in result["entries"][0]["reasons"][0]
    assert "did not pass the requested 24h horizon" in result["entries"][1]["reasons"][0]
    assert "not a factor-mining card" in result["entries"][2]["reasons"][0]
    assert "card id is missing" in result["entries"][3]["reasons"][0]
    assert result["counts"] == {
        "files": 5,
        "admitted": 1,
        "rejected": 3,
        "error": 1,
        "duplicate": 0,
        "groups": 1,
        "runner_representatives": 1,
    }
    assert result["entries"][4]["card_snapshot"]["id"] == "fmv6-valid"
    assert len(result["entries"][4]["card_sha256"]) == 64
    valid_path = tmp_path / "z-valid.json"
    assert result["entries"][4]["card_sha256"] == hashlib.sha256(valid_path.read_bytes()).hexdigest()
    assert result["entries"][4]["content_sha256"]


def test_scan_groups_fields_deduplicates_exact_signals_and_selects_structure_representative(tmp_path):
    _card(
        tmp_path / "later-id.json",
        card_id="fmv6-002",
        expression="ts_mean(perp_close, 4)",
    )
    _card(
        tmp_path / "earlier-id.json",
        card_id="fmv6-001",
        expression="ts_mean( perp_close, 4 )",
    )
    _card(
        tmp_path / "short-window.json",
        card_id="fmv6-003",
        expression="ts_mean(perp_close, 2)",
    )
    _card(
        tmp_path / "different-constant.json",
        card_id="fmv6-004",
        expression="mul(ts_mean(perp_close, 4), 2)",
    )
    _card(
        tmp_path / "opposite-direction.json",
        card_id="fmv6-005",
        expression="ts_mean(perp_close, 2)",
        direction=-1,
    )
    _card(
        tmp_path / "other-fields.json",
        card_id="fmv6-006",
        expression="ts_mean(spot_close, 2)",
    )

    result = scan_idea_pool(tmp_path, 24, 12, {"perp_close", "spot_close"})

    assert result["counts"]["admitted"] == 6
    groups = {(group["horizon_hours"], tuple(group["fields"])): group for group in result["groups"]}
    perp_group = groups[(24, ("perp_close",))]
    assert len(perp_group["members"]) == 5
    assert len(perp_group["exact_signals"]) == 4
    repeated_signal = next(
        signal for signal in perp_group["exact_signals"]
        if signal["expanded_expression"] == "ts_mean(perp_close, 4)"
    )
    assert [member["id"] for member in repeated_signal["members"]] == ["fmv6-001", "fmv6-002"]
    assert repeated_signal["representative"]["id"] == "fmv6-001"

    matching_family = next(
        family for family in perp_group["structure_families"]
        if {member["id"] for member in family["members"]} == {"fmv6-001", "fmv6-002", "fmv6-003"}
    )
    assert matching_family["kind"] == "structure_family"
    assert matching_family["family_id"]
    assert matching_family["representative"]["id"] == "fmv6-003"
    assert matching_family["representative_id"] == "fmv6-003"
    assert matching_family["member_ids"] == ["fmv6-003", "fmv6-001", "fmv6-002"]
    assert len(matching_family["member_paths"]) == 3
    family_ids = [
        family["family_id"] for group in result["groups"]
        for family in group["structure_families"]
    ]
    assert len(family_ids) == len(set(family_ids))
    assert len(perp_group["runner_representatives"]) == 3
    assert groups[(24, ("spot_close",))]["runner_representatives"][0]["id"] == "fmv6-006"


def test_scan_rejects_cards_requiring_unavailable_panel_fields(tmp_path):
    _card(
        tmp_path / "needs-spot.json",
        card_id="fmv6-spot",
        expression="spot_close",
    )

    result = scan_idea_pool(tmp_path, 24, 12, {"perp_close"})

    assert result["entries"][0]["status"] == "rejected"
    assert result["entries"][0]["reasons"] == [
        "required input fields are unavailable: ['spot_close']"
    ]
    assert result["groups"] == []


def test_scan_keeps_only_the_first_copy_of_an_identical_card_id(tmp_path):
    card = _card(tmp_path / "a-first.json", card_id="fmv6-same")
    (tmp_path / "b-copy.json").write_text(
        json.dumps(card, separators=(",", ":")), encoding="utf-8",
    )

    result = scan_idea_pool(tmp_path, 24, 12, {"perp_close"})

    assert [entry["status"] for entry in result["entries"]] == ["admitted", "duplicate"]
    assert result["entries"][1]["duplicate_of_path"] == result["entries"][0]["path"]
    assert result["entries"][0]["card_sha256"] != result["entries"][1]["card_sha256"]
    assert result["entries"][0]["content_sha256"] == result["entries"][1]["content_sha256"]
    for entry in result["entries"]:
        assert entry["card_sha256"] == hashlib.sha256(Path(entry["path"]).read_bytes()).hexdigest()
    assert result["counts"]["admitted"] == 1
    assert result["counts"]["duplicate"] == 1


def test_scan_marks_all_files_for_a_conflicting_card_id_as_errors(tmp_path):
    _card(tmp_path / "a-card.json", card_id="fmv6-conflict", expression="perp_close")
    _card(tmp_path / "b-card.json", card_id="fmv6-conflict", expression="spot_close")

    result = scan_idea_pool(tmp_path, 24, 12, {"perp_close", "spot_close"})

    assert [entry["status"] for entry in result["entries"]] == ["error", "error"]
    assert all("conflicting card contents share id fmv6-conflict" in entry["reasons"][-1]
               for entry in result["entries"])
    assert result["groups"] == []


def test_exact_signal_and_structure_family_ids_are_stable_across_horizons(tmp_path):
    _card(
        tmp_path / "multi-horizon.json",
        card_id="fmv6-multi-horizon",
        expression="ts_mean(perp_close, 4)",
        passed_horizons=[1, 4, 24],
    )

    one_hour = scan_idea_pool(tmp_path, 1, 12, {"perp_close"})
    one_day = scan_idea_pool(tmp_path, 24, 12, {"perp_close"})

    family_1h = one_hour["groups"][0]["structure_families"][0]
    family_24h = one_day["groups"][0]["structure_families"][0]
    signal_1h = one_hour["groups"][0]["exact_signals"][0]
    signal_24h = one_day["groups"][0]["exact_signals"][0]
    assert family_1h["family_id"] == family_24h["family_id"]
    assert signal_1h["key"] == signal_24h["key"]
