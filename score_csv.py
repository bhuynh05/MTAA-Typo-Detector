"""
Run the MTAA scorer over a CSV of participant transcriptions.

Usage:
    python score_csv.py input.csv output.csv \
        --target-col "sentence" --response-col "response"

The input CSV needs (at minimum) one column with the actual stimulus
sentence and one column with what the participant typed. Add
--target-col / --response-col if your headers differ from the defaults
below.

Output: the original CSV plus these new columns:
    overall_decision              accept / reject for the whole sentence
    needs_review                  True if any word landed on a low-confidence rule
    has_accepted_variation        True if sentence was accepted but contained non-exact matches
    accepted_variation_target     The target token(s) that were accepted despite a typo
    accepted_variation_response   The model's accepted typo token(s)
    accepted_variation_rule       The specific rule that permitted the accepted typo
    word_details                  JSON list of per-word decisions (target, response,
                                  decision, rule) for auditing / spot-checking
"""

import argparse
import csv
import json
import sys

from mtaa_scorer import score_sentence


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_csv")
    ap.add_argument("output_csv")
    ap.add_argument("--target-col", default="target_sentence")
    ap.add_argument("--response-col", default="response")
    args = ap.parse_args()

    with open(args.input_csv, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    if args.target_col not in fieldnames or args.response_col not in fieldnames:
        print(f"Columns found: {fieldnames}", file=sys.stderr)
        print(
            f"Could not find '{args.target_col}' and/or '{args.response_col}'. "
            f"Pass --target-col / --response-col with your actual header names.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Added the rule column and removed the final_word columns
    out_fieldnames = fieldnames + [
        "overall_decision", "needs_review", "has_accepted_variation", 
        "accepted_variation_target", "accepted_variation_response", "accepted_variation_rule",
        "word_details"
    ]
    n_review = 0
    n_reject = 0

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fieldnames)
        writer.writeheader()
        for row in rows:
            target = row.get(args.target_col, "") or ""
            response = row.get(args.response_col, "") or ""
            result = score_sentence(target, response)
            
            row["overall_decision"] = result["overall"]
            row["needs_review"] = result["needs_review"]
            row["has_accepted_variation"] = result["has_accepted_variation"]
            row["word_details"] = json.dumps(result["words"], ensure_ascii=False)

            # --- Isolate accepted variations and their rules ---
            var_targets = []
            var_responses = []
            var_rules = []
            
            if result["has_accepted_variation"]:
                for w in result["words"]:
                    # An accepted variation is any accepted word that isn't an exact match
                    if w["decision"] == "accept" and w["rule"] != "exact_match":
                        var_targets.append(str(w["target"]))
                        var_responses.append(str(w["response"]))
                        var_rules.append(str(w["rule"]))
            
            row["accepted_variation_target"] = " | ".join(var_targets)
            row["accepted_variation_response"] = " | ".join(var_responses)
            row["accepted_variation_rule"] = " | ".join(var_rules)
            # ---------------------------------------------------

            if result["overall"] == "reject":
                n_reject += 1
            if result["needs_review"]:
                n_review += 1
                
            writer.writerow(row)

    print(f"Scored {len(rows)} rows -> {args.output_csv}")
    print(f"  {n_reject} rejected overall (whole sentence)")
    print(f"  {n_review} flagged needs_review=True (low-confidence rule fired; spot-check these)")


if __name__ == "__main__":
    main()