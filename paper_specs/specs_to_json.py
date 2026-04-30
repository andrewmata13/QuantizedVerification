"""Parse all paper specs into a summary JSON."""
import os, re, json

SPEC_BASE = os.path.dirname(os.path.abspath(__file__))
ENVS = ["HalfCheetah-v4", "Hopper-v5"]

def parse_spec(path):
    with open(path) as f:
        text = f.read()

    comments = [l.lstrip("; ").strip() for l in text.splitlines() if l.startswith(";")]

    box_match = re.search(r"Box:\s*(\d+)%", "\n".join(comments))
    box_pct = int(box_match.group(1)) if box_match else None

    viol_match = re.search(r"Violation:\s*(.+)", "\n".join(comments))
    violation = viol_match.group(1).strip() if viol_match else None

    label_match = re.search(r"—\s*(.+)", comments[0]) if comments else None
    label = label_match.group(1).strip() if label_match else None

    bounds = {}
    for m in re.finditer(r"\(assert\s+\((>=|<=)\s+X_(\d+)\s+([^\)]+)\)\)", text):
        op, idx, val = m.group(1), int(m.group(2)), float(m.group(3))
        bounds.setdefault(idx, {})
        if op == ">=":
            bounds[idx]["lo"] = val
        else:
            bounds[idx]["hi"] = val

    input_box = {f"X_{i}": [bounds[i]["lo"], bounds[i]["hi"]]
                 for i in sorted(bounds)}

    output_clauses = []
    for m in re.finditer(r"\(and\s+\((>=|<=)\s+Y_(\d+)\s+([^\)]+)\)\)", text):
        op, idx, val = m.group(1), int(m.group(2)), float(m.group(3))
        output_clauses.append(f"Y_{idx} {op} {val}")

    return {
        "label": label,
        "box_pct": box_pct,
        "output_constraint": " OR ".join(output_clauses) if len(output_clauses) > 1 else output_clauses[0] if output_clauses else violation,
    }

specs = {}
for env in ENVS:
    env_dir = os.path.join(SPEC_BASE, env)
    if not os.path.isdir(env_dir):
        continue
    files = sorted(
        [f for f in os.listdir(env_dir) if f.endswith(".vnnlib")],
        key=lambda f: int(re.search(r"(\d+)", f).group(1))
    )
    for f in files:
        spec = parse_spec(os.path.join(env_dir, f))
        name = f.replace(".vnnlib", "")
        specs.setdefault(env, {})[name] = spec

out_path = os.path.join(SPEC_BASE, "specs_summary.json")
with open(out_path, "w") as f:
    json.dump(specs, f, indent=2)
print(json.dumps(specs, indent=2))
print(f"\nWritten to {out_path}")
