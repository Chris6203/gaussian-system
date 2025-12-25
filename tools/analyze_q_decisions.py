"""
Analyze Q-scorer decision records to understand why so few trades are happening.
"""
import json
import sys
from pathlib import Path
from collections import Counter, defaultdict

def analyze_decisions(decision_file):
    """Analyze decision records to find why trades are rejected."""

    decisions = []
    with open(decision_file, 'r') as f:
        for line in f:
            if line.strip():
                decisions.append(json.loads(line))

    print(f"\n{'='*80}")
    print(f"Q-SCORER DECISION ANALYSIS")
    print(f"{'='*80}\n")
    print(f"Total decisions: {len(decisions)}")

    # Count actions
    actions = Counter(d.get('best_action', 'UNKNOWN') for d in decisions)
    print(f"\nAction Distribution:")
    for action, count in actions.most_common():
        pct = 100 * count / len(decisions)
        print(f"  {action:10s}: {count:5d} ({pct:5.1f}%)")

    # Analyze Q-values
    q_holds = [d.get('q_hold', 0) for d in decisions if 'q_hold' in d]
    q_calls = [d.get('q_call', 0) for d in decisions if 'q_call' in d]
    q_puts = [d.get('q_put', 0) for d in decisions if 'q_put' in d]

    if q_holds:
        print(f"\nQ-Value Statistics:")
        print(f"  Q_HOLD: min={min(q_holds):.2f}, max={max(q_holds):.2f}, avg={sum(q_holds)/len(q_holds):.2f}")
    if q_calls:
        print(f"  Q_CALL: min={min(q_calls):.2f}, max={max(q_calls):.2f}, avg={sum(q_calls)/len(q_calls):.2f}")
    if q_puts:
        print(f"  Q_PUT:  min={min(q_puts):.2f}, max={max(q_puts):.2f}, avg={sum(q_puts)/len(q_puts):.2f}")

    # Analyze thresholds/filters
    entries_attempted = sum(1 for d in decisions if d.get('best_action') in ['CALL', 'PUT'])
    entries_passed_threshold = sum(1 for d in decisions if d.get('passed_threshold', False))
    entries_passed_conf = sum(1 for d in decisions if d.get('passed_confidence', False))
    entries_passed_ret = sum(1 for d in decisions if d.get('passed_return', False))

    print(f"\nEntry Filter Analysis (of {entries_attempted} entry signals):")
    if entries_attempted > 0:
        print(f"  Passed Q-threshold:  {entries_passed_threshold:5d} ({100*entries_passed_threshold/entries_attempted:.1f}%)")
        print(f"  Passed confidence:   {entries_passed_conf:5d} ({100*entries_passed_conf/entries_attempted:.1f}%)")
        print(f"  Passed min return:   {entries_passed_ret:5d} ({100*entries_passed_ret/entries_attempted:.1f}%)")

    # Analyze by regime
    regime_actions = defaultdict(Counter)
    for d in decisions:
        regime = d.get('regime', 'unknown')
        action = d.get('best_action', 'UNKNOWN')
        regime_actions[regime][action] += 1

    print(f"\nAction by Regime:")
    for regime in sorted(regime_actions.keys()):
        total = sum(regime_actions[regime].values())
        print(f"  {regime:15s} (n={total:4d}):", end='')
        for action in ['HOLD', 'CALL', 'PUT']:
            count = regime_actions[regime][action]
            if count > 0:
                print(f" {action}={count}({100*count/total:.0f}%)", end='')
        print()

    # Find sample trades that WERE taken
    trades_taken = [d for d in decisions if d.get('best_action') in ['CALL', 'PUT']
                    and d.get('passed_threshold') and d.get('passed_confidence')]

    print(f"\nSample Trades That PASSED Filters ({len(trades_taken)} total):")
    for i, d in enumerate(trades_taken[:5]):
        print(f"\n  Trade {i+1}:")
        print(f"    Action: {d.get('best_action')}")
        print(f"    Q-values: HOLD={d.get('q_hold', 0):.2f}, CALL={d.get('q_call', 0):.2f}, PUT={d.get('q_put', 0):.2f}")
        print(f"    Regime: {d.get('regime', 'unknown')}")
        print(f"    Timestamp: {d.get('timestamp', 'unknown')}")

    # Find sample trades that were REJECTED
    trades_rejected = [d for d in decisions if d.get('best_action') in ['CALL', 'PUT']
                      and not (d.get('passed_threshold') and d.get('passed_confidence'))]

    print(f"\nSample Trades That FAILED Filters ({len(trades_rejected)} total):")
    for i, d in enumerate(trades_rejected[:5]):
        print(f"\n  Rejected {i+1}:")
        print(f"    Action: {d.get('best_action')}")
        print(f"    Q-values: HOLD={d.get('q_hold', 0):.2f}, CALL={d.get('q_call', 0):.2f}, PUT={d.get('q_put', 0):.2f}")
        print(f"    Passed threshold: {d.get('passed_threshold', False)}")
        print(f"    Passed confidence: {d.get('passed_confidence', False)}")
        print(f"    Regime: {d.get('regime', 'unknown')}")

    print(f"\n{'='*80}\n")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python analyze_q_decisions.py <path_to_decision_records.jsonl>")
        sys.exit(1)

    decision_file = Path(sys.argv[1])
    if not decision_file.exists():
        print(f"Error: File not found: {decision_file}")
        sys.exit(1)

    analyze_decisions(decision_file)
