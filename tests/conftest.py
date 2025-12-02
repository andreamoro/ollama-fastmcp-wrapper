"""Pytest configuration and fixtures for test result tracking."""

import pytest
import json
from pathlib import Path
from datetime import datetime
import shutil


# Global test results collector
test_results = []
test_start_time = None


@pytest.hookimpl(tryfirst=True)
def pytest_sessionstart(session):
    """Called before test session starts."""
    global test_start_time
    test_start_time = datetime.now()

    # Clear previous results
    test_results.clear()

    # Create test_results directory
    results_dir = Path("tests/test_results")
    results_dir.mkdir(exist_ok=True)

    print(f"\n📊 Test results will be saved to: {results_dir}")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Hook to capture test results."""
    outcome = yield
    report = outcome.get_result()

    if report.when == "call":
        test_results.append({
            "name": item.nodeid,
            "outcome": report.outcome,
            "duration": report.duration,
            "markers": [m.name for m in item.iter_markers()],
        })


def pytest_sessionfinish(session, exitstatus):
    """Called after test session finishes."""
    results_dir = Path("tests/test_results")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Calculate statistics
    passed = sum(1 for r in test_results if r["outcome"] == "passed")
    failed = sum(1 for r in test_results if r["outcome"] == "failed")
    skipped = sum(1 for r in test_results if r["outcome"] == "skipped")
    total = len(test_results)
    total_duration = sum(r["duration"] for r in test_results)

    # Save detailed JSON report
    report = {
        "timestamp": timestamp,
        "session_start": test_start_time.isoformat(),
        "session_end": datetime.now().isoformat(),
        "summary": {
            "total": total,
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "duration_seconds": round(total_duration, 2),
        },
        "tests": test_results,
        "exit_status": exitstatus,
    }

    json_file = results_dir / f"test_report_{timestamp}.json"
    with open(json_file, "w") as f:
        json.dump(report, f, indent=2)

    # Generate markdown report
    md_content = generate_markdown_report(report)
    md_file = results_dir / f"test_report_{timestamp}.md"
    with open(md_file, "w") as f:
        f.write(md_content)

    # Create/update latest.md symlink
    latest_md = results_dir / "latest.md"
    if latest_md.exists():
        latest_md.unlink()
    shutil.copy(md_file, latest_md)

    print(f"\n✅ Test report saved:")
    print(f"   📄 {json_file}")
    print(f"   📝 {md_file}")
    print(f"   🔗 {latest_md}")


def generate_markdown_report(report):
    """Generate a markdown report from test results."""
    summary = report["summary"]

    # Determine status emoji
    if summary["failed"] > 0:
        status_emoji = "❌"
        status_text = "FAILED"
    elif summary["skipped"] == summary["total"]:
        status_emoji = "⏭️"
        status_text = "ALL SKIPPED"
    else:
        status_emoji = "✅"
        status_text = "PASSED"

    md = f"""# Test Report - {status_emoji} {status_text}

**Generated:** {report["timestamp"]}
**Duration:** {summary["duration_seconds"]}s

## Summary

| Status | Count |
|--------|-------|
| ✅ Passed | {summary["passed"]} |
| ❌ Failed | {summary["failed"]} |
| ⏭️ Skipped | {summary["skipped"]} |
| **Total** | **{summary["total"]}** |

## Test Results

"""

    # Group tests by outcome
    passed_tests = [t for t in report["tests"] if t["outcome"] == "passed"]
    failed_tests = [t for t in report["tests"] if t["outcome"] == "failed"]
    skipped_tests = [t for t in report["tests"] if t["outcome"] == "skipped"]

    if passed_tests:
        md += "### ✅ Passed Tests\n\n"
        for test in passed_tests:
            markers = f" `[{', '.join(test['markers'])}]`" if test['markers'] else ""
            md += f"- `{test['name']}`{markers} ({test['duration']:.3f}s)\n"
        md += "\n"

    if failed_tests:
        md += "### ❌ Failed Tests\n\n"
        for test in failed_tests:
            markers = f" `[{', '.join(test['markers'])}]`" if test['markers'] else ""
            md += f"- `{test['name']}`{markers} ({test['duration']:.3f}s)\n"
        md += "\n"

    if skipped_tests:
        md += "### ⏭️ Skipped Tests\n\n"
        for test in skipped_tests:
            markers = f" `[{', '.join(test['markers'])}]`" if test['markers'] else ""
            md += f"- `{test['name']}`{markers}\n"
        md += "\n"

    # Add timing analysis
    md += "## Timing Analysis\n\n"
    sorted_tests = sorted(report["tests"], key=lambda x: x["duration"], reverse=True)
    top_5 = sorted_tests[:5]

    md += "**Slowest Tests:**\n\n"
    for i, test in enumerate(top_5, 1):
        md += f"{i}. `{test['name'].split('::')[-1]}` - {test['duration']:.3f}s\n"

    md += f"\n**Average test duration:** {summary['duration_seconds'] / max(summary['total'], 1):.3f}s\n"

    return md


@pytest.fixture
def test_results_dir():
    """Fixture to provide test results directory for tests that need to save files."""
    results_dir = Path("tests/test_results")
    results_dir.mkdir(exist_ok=True)
    return results_dir
