"""Create an offline HTML comparison; aggregate seeds only within identical protocols."""
import argparse
import csv
import html
import json
import statistics
from collections import defaultdict
from pathlib import Path


def chart(history):
    values = [r[k] for r in history for k in ('train_loss', 'val_loss')]
    maximum = max(values + [.01])
    paths = []
    for key, color in [('train_loss', '#00b9a7'), ('val_loss', '#8c7bff')]:
        points = ' '.join(f'{35+i*410/max(len(history)-1,1):.1f},{175-r[key]/maximum*145:.1f}' for i,r in enumerate(history))
        paths.append(f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{points}"/>')
    return '<svg viewBox="0 0 480 210" role="img" aria-label="train and validation loss by epoch"><path d="M35 20V175H455" stroke="#789" fill="none"/>' + ''.join(paths) + '<text x="40" y="200" fill="#00b9a7">train loss</text><text x="230" y="200" fill="#8c7bff">validation loss</text></svg>'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='runs')
    args = p.parse_args()
    root = Path(args.root)
    root.mkdir(exist_ok=True, parents=True)
    groups = defaultdict(list)
    rows = []
    for path in sorted(root.glob('*/result.json')):
        row = json.loads(path.read_text(encoding='utf-8'))
        row['run'] = path.parent.name
        rows.append(row)
        groups[(row['task'], row['comparison_group'])].append(row)
    output = ['<!doctype html><html lang="ko"><meta charset="utf-8"><title>AI Lab 결과</title>',
              '<style>body{font:16px system-ui;background:#101824;color:#edf3ff;max-width:1200px;margin:40px auto;padding:20px}table{border-collapse:collapse;width:100%}td,th{padding:12px;border-bottom:1px solid #456;text-align:left}article{background:#192637;padding:20px;margin:18px 0;border-radius:14px}svg{max-width:480px}a{color:#74e2d6}</style>',
              '<h1>실제로 측정한 모델 비교</h1><p>정확도 ↑ · loss/perplexity ↓. 서로 다른 task/group은 직접 순위를 매기지 마세요. quick은 기능 점검입니다.</p>',
              '<p>시간은 학습+매 epoch validation+체크포인트 저장을 포함합니다. peak CUDA allocated는 PyTorch tensor allocation이며 GPU 전체 점유량/전력이 아닙니다.</p>']
    if not rows:
        output.append('<p>아직 결과가 없습니다. python -m ailab.run 을 먼저 실행하세요.</p>')
    for (task, group), runs in groups.items():
        baseline_name = {'digits':'dnn','recall':'rnn','language':'lstm'}[task]
        baseline_items = {r['seed']:r for r in runs if r['model']==baseline_name}
        baseline = statistics.mean(r['test_accuracy'] for r in baseline_items.values()) if baseline_items else None
        output.append(f'<h2>{html.escape(task)} · group {group}</h2><table><tr><th>model</th><th>seeds (n)</th><th>test accuracy mean±SD</th><th>Δ accuracy (pp)</th><th>test loss</th><th>PPL</th><th>params</th><th>train seconds</th><th>peak MB</th></tr>')
        by_model = defaultdict(list)
        # Re-running the same seed must not artificially inflate sample count.
        for row in runs:
            by_model[row['model']].append(row)
        for model, items in by_model.items():
            latest = {r['seed']: r for r in items}
            items = list(latest.values())
            accuracies = [r['test_accuracy'] for r in items]
            sd = statistics.stdev(accuracies) if len(items) > 1 else None
            acc = f'{statistics.mean(accuracies)*100:.2f}%' + (f' ± {sd*100:.2f} pp' if sd is not None else ' (SD 미측정)')
            mean = lambda k: statistics.mean(r[k] for r in items)
            ppl = f"{mean('perplexity'):.3f}" if task == 'language' else '—'
            memory = f"{mean('peak_cuda_allocated_mb'):.1f}" if items[0]['peak_cuda_allocated_mb'] is not None else 'CPU'
            delta = f'{(statistics.mean(accuracies)-baseline)*100:+.2f}' if baseline is not None else '—'
            output.append(f'<tr><td>{html.escape(model)}</td><td>{sorted(latest)} ({len(items)})</td><td>{acc}</td><td>{delta}</td><td>{mean("test_loss"):.4f}</td><td>{ppl}</td><td>{items[0]["parameters"]:,}</td><td>{mean("training_seconds"):.2f}</td><td>{memory}</td></tr>')
        output.append(f'</table><p>Δ baseline: {baseline_name}. Accuracy percentage points, not relative percent.</p>')
        for row in runs:
            output.append(f'<article><h3>{html.escape(row["run"])}</h3><p>{"QUICK / 기능 점검" if row["quick"] else "FULL"} · best epoch {row["best_epoch"]} · {html.escape(row["device"])}</p>{chart(row["history"])}</article>')
    output.append('</html>')
    (root/'comparison.html').write_text('\n'.join(output), encoding='utf-8')
    if rows:
        keys = [k for k in rows[0] if k != 'history']
        with (root/'summary.csv').open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)
    print('Open:', (root/'comparison.html').resolve())


if __name__ == '__main__':
    main()
