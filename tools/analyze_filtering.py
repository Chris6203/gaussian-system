"""
Analyze tradability gate dataset to understand filtering behavior and missed opportunities.
"""
import json
import sys
from pathlib import Path
from collections import Counter
import statistics

def analyze_filtering(dataset_file):
    """Analyze filtering decisions and missed opportunities."""

    records = []
    with open(dataset_file, 'r') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    print(f"\n{'='*80}")
    print(f"FILTERING ANALYSIS")
    print(f"{'='*80}\n")
    print(f"Total opportunities: {len(records)}")

    # Separate by label
    taken = [r for r in records if r.get('label') == 1]
    rejected = [r for r in records if r.get('label') == 0]

    print(f"\nEntry Rate:")
    print(f"  Trades TAKEN:    {len(taken):5d} ({100*len(taken)/len(records):.1f}%)")
    print(f"  Trades REJECTED: {len(rejected):5d} ({100*len(rejected)/len(records):.1f}%)")

    # Analyze confidence distributions
    if taken:
        taken_conf = [r['features']['prediction_confidence'] for r in taken]
        taken_ret = [r['features']['predicted_return'] for r in taken]
        taken_realized = [r['realized_move'] for r in taken]

        print(f"\n{'='*80}")
        print(f"TRADES THAT WERE TAKEN - {len(taken)} total")
        print(f"{'='*80}")

        print(f"\nPrediction Confidence:")
        print(f"  Min: {min(taken_conf):.4f}, Max: {max(taken_conf):.4f}")
        print(f"  Mean: {statistics.mean(taken_conf):.4f}, Median: {statistics.median(taken_conf):.4f}")

        print(f"\nPredicted Returns:")
        print(f"  Min: {min(taken_ret):.6f}, Max: {max(taken_ret):.6f}")
        print(f"  Mean: {statistics.mean(taken_ret):.6f}, Median: {statistics.median(taken_ret):.6f}")

        print(f"\nRealized Returns:")
        print(f"  Min: {min(taken_realized):.6f}, Max: {max(taken_realized):.6f}")
        print(f"  Mean: {statistics.mean(taken_realized):.6f}, Median: {statistics.median(taken_realized):.6f}")

        # Win rate
        winners = [r for r in taken_realized if r > 0]
        losers = [r for r in taken_realized if r < 0]
        print(f"\nWin Rate:")
        print(f"  Winners: {len(winners)} ({100*len(winners)/len(taken):.1f}%)")
        print(f"  Losers:  {len(losers)} ({100*len(losers)/len(taken):.1f}%)")

    if rejected:
        rej_conf = [r['features']['prediction_confidence'] for r in rejected]
        rej_ret = [r['features']['predicted_return'] for r in rejected]
        rej_realized = [r['realized_move'] for r in rejected]

        print(f"\n{'='*80}")
        print(f"TRADES THAT WERE REJECTED - {len(rejected)} total")
        print(f"{'='*80}")

        print(f"\nPrediction Confidence:")
        print(f"  Min: {min(rej_conf):.4f}, Max: {max(rej_conf):.4f}")
        print(f"  Mean: {statistics.mean(rej_conf):.4f}, Median: {statistics.median(rej_conf):.4f}")

        print(f"\nPredicted Returns:")
        print(f"  Min: {min(rej_ret):.6f}, Max: {max(rej_ret):.6f}")
        print(f"  Mean: {statistics.mean(rej_ret):.6f}, Median: {statistics.median(rej_ret):.6f}")

        print(f"\nRealized Returns (what WOULD have happened):")
        print(f"  Min: {min(rej_realized):.6f}, Max: {max(rej_realized):.6f}")
        print(f"  Mean: {statistics.mean(rej_realized):.6f}, Median: {statistics.median(rej_realized):.6f}")

        # What we missed
        would_win = [r for r in rej_realized if r > 0]
        would_lose = [r for r in rej_realized if r < 0]
        print(f"\nWhat We Missed:")
        print(f"  Would-be winners: {len(would_win)} ({100*len(would_win)/len(rejected):.1f}%)")
        print(f"  Would-be losers:  {len(would_lose)} ({100*len(would_lose)/len(rejected):.1f}%)")

        # Find high-confidence rejected trades
        high_conf_rejected = [r for r in rejected if r['features']['prediction_confidence'] > 0.3]

        if high_conf_rejected:
            print(f"\n{'='*80}")
            print(f"HIGH-CONFIDENCE REJECTED (conf > 0.3) - {len(high_conf_rejected)} total")
            print(f"{'='*80}")
            print("These had high confidence but were still filtered out.\n")

            hc_realized = [r['realized_move'] for r in high_conf_rejected]
            hc_winners = [r for r in hc_realized if r > 0]
            hc_losers = [r for r in hc_realized if r < 0]

            print(f"Would-be Performance:")
            print(f"  Winners: {len(hc_winners)} ({100*len(hc_winners)/len(high_conf_rejected):.1f}%)")
            print(f"  Losers:  {len(hc_losers)} ({100*len(hc_losers)/len(high_conf_rejected):.1f}%)")
            print(f"  Mean return: {statistics.mean(hc_realized):.6f}")

            # Show top missed opportunities
            high_conf_rejected_sorted = sorted(high_conf_rejected,
                                              key=lambda x: x['realized_move'],
                                              reverse=True)

            print(f"\nTop 10 MISSED WINNERS (high confidence, rejected):")
            for i, rec in enumerate(high_conf_rejected_sorted[:10]):
                if rec['realized_move'] > 0:
                    print(f"\n  Missed {i+1}:")
                    print(f"    Timestamp: {rec['timestamp']}")
                    print(f"    Action: {rec['proposed_action']}")
                    print(f"    Confidence: {rec['features']['prediction_confidence']:.4f}")
                    print(f"    Predicted: {rec['features']['predicted_return']:.6f}")
                    print(f"    Realized: {rec['realized_move']:.6f} >>> WOULD HAVE WON!")
                    print(f"    Reason rejected: {rec.get('reason', 'unknown')}")

        # Find low-confidence that would have won
        low_conf_winners = [r for r in rejected
                           if r['features']['prediction_confidence'] < 0.2
                           and r['realized_move'] > 0.01]

        if low_conf_winners:
            print(f"\n{'='*80}")
            print(f"LOW-CONFIDENCE BIG WINNERS (conf < 0.2, return > 1%) - {len(low_conf_winners)} total")
            print(f"{'='*80}")
            print("These were filtered due to low confidence but turned out to be big winners.\n")

            low_conf_winners_sorted = sorted(low_conf_winners,
                                            key=lambda x: x['realized_move'],
                                            reverse=True)

            for i, rec in enumerate(low_conf_winners_sorted[:5]):
                print(f"\n  Missed {i+1}:")
                print(f"    Timestamp: {rec['timestamp']}")
                print(f"    Action: {rec['proposed_action']}")
                print(f"    Confidence: {rec['features']['prediction_confidence']:.4f}")
                print(f"    Predicted: {rec['features']['predicted_return']:.6f}")
                print(f"    Realized: {rec['realized_move']:.6f} >>> HUGE WINNER!")
                print(f"    Reason rejected: {rec.get('reason', 'unknown')}")

    # Confidence threshold analysis
    print(f"\n{'='*80}")
    print(f"CONFIDENCE THRESHOLD ANALYSIS")
    print(f"{'='*80}\n")

    for threshold in [0.10, 0.15, 0.20, 0.25, 0.30]:
        above_thresh = [r for r in records if r['features']['prediction_confidence'] >= threshold]
        if above_thresh:
            realized = [r['realized_move'] for r in above_thresh]
            winners = [r for r in realized if r > 0]
            mean_return = statistics.mean(realized)

            print(f"Confidence >= {threshold:.2f}:")
            print(f"  Count: {len(above_thresh):4d} ({100*len(above_thresh)/len(records):.1f}% of all)")
            print(f"  Win rate: {100*len(winners)/len(above_thresh):.1f}%")
            print(f"  Mean return: {mean_return:.6f}")

    print(f"\n{'='*80}\n")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python analyze_filtering.py <path_to_tradability_gate_dataset.jsonl>")
        sys.exit(1)

    dataset_file = Path(sys.argv[1])
    if not dataset_file.exists():
        print(f"Error: File not found: {dataset_file}")
        sys.exit(1)

    analyze_filtering(dataset_file)
