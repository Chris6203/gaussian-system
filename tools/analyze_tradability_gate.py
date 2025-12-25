"""
Analyze tradability gate dataset to understand Q-value distributions and filtering behavior.
"""
import json
import sys
from pathlib import Path
from collections import Counter
import statistics

def analyze_tradability_gate(dataset_file):
    """Analyze tradability gate dataset."""

    records = []
    with open(dataset_file, 'r') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    print(f"\n{'='*80}")
    print(f"TRADABILITY GATE ANALYSIS")
    print(f"{'='*80}\n")
    print(f"Total records: {len(records)}")

    # Separate by label
    positives = [r for r in records if r.get('label') == 1]
    negatives = [r for r in records if r.get('label') == 0]

    print(f"\nLabel Distribution:")
    print(f"  Positive (tradable): {len(positives)} ({100*len(positives)/len(records):.1f}%)")
    print(f"  Negative (not tradable): {len(negatives)} ({100*len(negatives)/len(records):.1f}%)")

    # Analyze Q-values for positives vs negatives
    if positives:
        pos_q_call = [r.get('q_call', 0) for r in positives if 'q_call' in r]
        pos_q_put = [r.get('q_put', 0) for r in positives if 'q_put' in r]
        pos_q_hold = [r.get('q_hold', 0) for r in positives if 'q_hold' in r]

        print(f"\n{'='*80}")
        print(f"POSITIVE SAMPLES (Trades that WERE taken) - {len(positives)} total")
        print(f"{'='*80}")

        if pos_q_call:
            print(f"\nQ_CALL for positives:")
            print(f"  Min: {min(pos_q_call):.2f}, Max: {max(pos_q_call):.2f}")
            print(f"  Mean: {statistics.mean(pos_q_call):.2f}, Median: {statistics.median(pos_q_call):.2f}")
            print(f"  Stdev: {statistics.stdev(pos_q_call) if len(pos_q_call) > 1 else 0:.2f}")

        if pos_q_put:
            print(f"\nQ_PUT for positives:")
            print(f"  Min: {min(pos_q_put):.2f}, Max: {max(pos_q_put):.2f}")
            print(f"  Mean: {statistics.mean(pos_q_put):.2f}, Median: {statistics.median(pos_q_put):.2f}")
            print(f"  Stdev: {statistics.stdev(pos_q_put) if len(pos_q_put) > 1 else 0:.2f}")

        if pos_q_hold:
            print(f"\nQ_HOLD for positives:")
            print(f"  Min: {min(pos_q_hold):.2f}, Max: {max(pos_q_hold):.2f}")
            print(f"  Mean: {statistics.mean(pos_q_hold):.2f}, Median: {statistics.median(pos_q_hold):.2f}")
            print(f"  Stdev: {statistics.stdev(pos_q_hold) if len(pos_q_hold) > 1 else 0:.2f}")

        # Show sample positive records
        print(f"\nSample POSITIVE records (first 5):")
        for i, rec in enumerate(positives[:5]):
            print(f"\n  Sample {i+1}:")
            q_call_str = f"{rec['q_call']:.2f}" if 'q_call' in rec else 'N/A'
            q_put_str = f"{rec['q_put']:.2f}" if 'q_put' in rec else 'N/A'
            q_hold_str = f"{rec['q_hold']:.2f}" if 'q_hold' in rec else 'N/A'
            print(f"    Q_CALL: {q_call_str}")
            print(f"    Q_PUT:  {q_put_str}")
            print(f"    Q_HOLD: {q_hold_str}")
            print(f"    Action: {rec.get('action', 'N/A')}")
            print(f"    Future Return: {rec.get('future_return', 'N/A')}")

    if negatives:
        neg_q_call = [r.get('q_call', 0) for r in negatives if 'q_call' in r]
        neg_q_put = [r.get('q_put', 0) for r in negatives if 'q_put' in r]
        neg_q_hold = [r.get('q_hold', 0) for r in negatives if 'q_hold' in r]

        print(f"\n{'='*80}")
        print(f"NEGATIVE SAMPLES (Trades that were REJECTED) - {len(negatives)} total")
        print(f"{'='*80}")

        if neg_q_call:
            print(f"\nQ_CALL for negatives:")
            print(f"  Min: {min(neg_q_call):.2f}, Max: {max(neg_q_call):.2f}")
            print(f"  Mean: {statistics.mean(neg_q_call):.2f}, Median: {statistics.median(neg_q_call):.2f}")
            print(f"  Stdev: {statistics.stdev(neg_q_call) if len(neg_q_call) > 1 else 0:.2f}")

        if neg_q_put:
            print(f"\nQ_PUT for negatives:")
            print(f"  Min: {min(neg_q_put):.2f}, Max: {max(neg_q_put):.2f}")
            print(f"  Mean: {statistics.mean(neg_q_put):.2f}, Median: {statistics.median(neg_q_put):.2f}")
            print(f"  Stdev: {statistics.stdev(neg_q_put) if len(neg_q_put) > 1 else 0:.2f}")

        if neg_q_hold:
            print(f"\nQ_HOLD for negatives:")
            print(f"  Min: {min(neg_q_hold):.2f}, Max: {max(neg_q_hold):.2f}")
            print(f"  Mean: {statistics.mean(neg_q_hold):.2f}, Median: {statistics.median(neg_q_hold):.2f}")
            print(f"  Stdev: {statistics.stdev(neg_q_hold) if len(neg_q_hold) > 1 else 0:.2f}")

        # Find high-confidence rejected trades (potential missed opportunities)
        high_conf_rejected = []
        for rec in negatives:
            q_call = rec.get('q_call', float('-inf'))
            q_put = rec.get('q_put', float('-inf'))
            q_hold = rec.get('q_hold', 0)

            max_q = max(q_call, q_put, q_hold)
            if max_q > 20:  # High Q-value but was rejected
                margin = max_q - q_hold
                high_conf_rejected.append({
                    'rec': rec,
                    'margin': margin,
                    'max_q': max_q
                })

        if high_conf_rejected:
            high_conf_rejected.sort(key=lambda x: x['margin'], reverse=True)
            print(f"\n{'='*80}")
            print(f"HIGH-CONFIDENCE REJECTED TRADES (Q > 20) - {len(high_conf_rejected)} total")
            print(f"{'='*80}")
            print("These are trades with strong Q-values that were filtered out.")
            print("If many of these would have been winners, filters are too strict.\n")

            for i, item in enumerate(high_conf_rejected[:10]):
                rec = item['rec']
                print(f"\nRejected {i+1} (margin={item['margin']:.2f}):")
                q_call_str = f"{rec['q_call']:.2f}" if 'q_call' in rec else 'N/A'
                q_put_str = f"{rec['q_put']:.2f}" if 'q_put' in rec else 'N/A'
                q_hold_str = f"{rec['q_hold']:.2f}" if 'q_hold' in rec else 'N/A'
                print(f"  Q_CALL: {q_call_str}")
                print(f"  Q_PUT:  {q_put_str}")
                print(f"  Q_HOLD: {q_hold_str}")
                print(f"  Action: {rec.get('action', 'N/A')}")
                print(f"  Future Return: {rec.get('future_return', 'N/A')}")

                # Determine if this would have been a winner
                fut_ret = rec.get('future_return')
                if fut_ret is not None:
                    if fut_ret > 0:
                        print(f"  >>> MISSED WINNER! (would have gained {fut_ret:.4f})")
                    else:
                        print(f"  >>> Correctly rejected loser (would have lost {fut_ret:.4f})")

        # Show sample negative records
        print(f"\nSample NEGATIVE records (first 5):")
        for i, rec in enumerate(negatives[:5]):
            print(f"\n  Sample {i+1}:")
            q_call_str = f"{rec['q_call']:.2f}" if 'q_call' in rec else 'N/A'
            q_put_str = f"{rec['q_put']:.2f}" if 'q_put' in rec else 'N/A'
            q_hold_str = f"{rec['q_hold']:.2f}" if 'q_hold' in rec else 'N/A'
            print(f"    Q_CALL: {q_call_str}")
            print(f"    Q_PUT:  {q_put_str}")
            print(f"    Q_HOLD: {q_hold_str}")
            print(f"    Action: {rec.get('action', 'N/A')}")
            print(f"    Future Return: {rec.get('future_return', 'N/A')}")

    # Overall Q-value comparison
    print(f"\n{'='*80}")
    print(f"Q-VALUE COMPARISON: Taken vs Rejected")
    print(f"{'='*80}")

    all_q_call = [r.get('q_call', 0) for r in records if 'q_call' in r]
    all_q_put = [r.get('q_put', 0) for r in records if 'q_put' in r]
    all_q_hold = [r.get('q_hold', 0) for r in records if 'q_hold' in r]

    if all_q_call:
        print(f"\nALL Q_CALL:")
        print(f"  Min: {min(all_q_call):.2f}, Max: {max(all_q_call):.2f}")
        print(f"  Mean: {statistics.mean(all_q_call):.2f}, Median: {statistics.median(all_q_call):.2f}")

    if all_q_put:
        print(f"\nALL Q_PUT:")
        print(f"  Min: {min(all_q_put):.2f}, Max: {max(all_q_put):.2f}")
        print(f"  Mean: {statistics.mean(all_q_put):.2f}, Median: {statistics.median(all_q_put):.2f}")

    if all_q_hold:
        print(f"\nALL Q_HOLD:")
        print(f"  Min: {min(all_q_hold):.2f}, Max: {max(all_q_hold):.2f}")
        print(f"  Mean: {statistics.mean(all_q_hold):.2f}, Median: {statistics.median(all_q_hold):.2f}")

    print(f"\n{'='*80}\n")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python analyze_tradability_gate.py <path_to_tradability_gate_dataset.jsonl>")
        sys.exit(1)

    dataset_file = Path(sys.argv[1])
    if not dataset_file.exists():
        print(f"Error: File not found: {dataset_file}")
        sys.exit(1)

    analyze_tradability_gate(dataset_file)
