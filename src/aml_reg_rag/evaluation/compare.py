from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from aml_reg_rag.evaluation.runner import EvalReport, retrieval_label

RETRIEVAL_METRICS = (
    ("Hit@1", "hit_at_1"),
    ("Hit@3", "hit_at_3"),
    ("Recall@k", "recall_at_k"),
    ("MRR", "mrr"),
)
GENERATION_METRICS = (
    ("驗證通過", "verified_rate"),
    ("拒答正確", "refusal_accuracy"),
    ("要點涵蓋", "coverage"),
    ("成本", "cost_usd"),
)


class ReportError(ValueError):
    pass


@dataclass(frozen=True)
class Comparison:
    columns: list[str]
    rows: list[tuple[str, list[str]]]


def load_report(path: Path) -> EvalReport:
    try:
        return EvalReport.model_validate_json(path.read_text(encoding="utf-8"))
    except (ValidationError, UnicodeDecodeError, OSError) as exc:
        raise ReportError(f"{path.name} 不是有效的 eval 結果檔") from exc


def report_label(report: EvalReport) -> str:
    retrieval = retrieval_label(report.settings)
    label = f"{retrieval} · {report.model}" if report.model else retrieval
    return f"{label}\n{report.created_at:%m-%d %H:%M}"


def compare_reports(reports: list[EvalReport]) -> Comparison:
    groups = list(dict.fromkeys(name for report in reports for name in report.retrieval))
    rows: list[tuple[str, list[str]]] = []
    for group in groups:
        for label, attr in RETRIEVAL_METRICS:
            values = []
            for report in reports:
                summary = report.retrieval.get(group)
                value = None if summary is None else getattr(summary, attr)
                values.append(_format(attr, value))
            rows.append((f"{group} {label}", values))
    if any(report.generation for report in reports):
        for label, attr in GENERATION_METRICS:
            values = []
            for report in reports:
                generation = report.generation.get("all")
                values.append(
                    _format(attr, None if generation is None else getattr(generation, attr))
                )
            rows.append((f"生成 {label}", values))
    return Comparison([report_label(report) for report in reports], rows)


def _format(attr: str, value: float | None) -> str:
    if value is None:
        return "—"
    if attr == "mrr":
        return f"{value:.3f}"
    if attr == "cost_usd":
        return f"${value:.4f}"
    return f"{value:.0%}"
