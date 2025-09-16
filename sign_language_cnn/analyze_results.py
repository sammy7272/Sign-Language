import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

METRICS_CSV = Path('results/experiments_metrics.csv')
PLOTS_DIR = Path('results/plots')


def main():
	if not METRICS_CSV.exists():
		print(f"Not found: {METRICS_CSV}")
		return
	df = pd.read_csv(METRICS_CSV)
	PLOTS_DIR.mkdir(parents=True, exist_ok=True)

	# Normalize column names present for per-class precision/recall and AUC
	# Expected columns: 'accuracy', 'precision_0'..'precision_9', 'recall_0'.., 'AUC_0'.. etc. (class names were digits)

	# Overall accuracy comparison by run
	plt.figure(figsize=(10, 4))
	sns.barplot(data=df.sort_values('accuracy', ascending=False), x='run_name', y='accuracy', color='#4c72b0')
	plt.xticks(rotation=60, ha='right')
	plt.title('Overall Accuracy by Run')
	plt.tight_layout()
	plt.savefig(PLOTS_DIR / 'cmp_overall_accuracy.png', dpi=150)
	plt.close()

	# Macro precision/recall/AUC if available; otherwise compute macro from per-class columns
	def macro_avg(prefix: str) -> pd.Series:
		cols = [c for c in df.columns if c.startswith(prefix + '_')]
		if not cols:
			return pd.Series([None] * len(df))
		return df[cols].mean(axis=1)

	df['macro_precision'] = macro_avg('precision')
	df['macro_recall'] = macro_avg('recall')
	df['macro_auc'] = macro_avg('AUC')

	for metric in ['macro_precision', 'macro_recall', 'macro_auc']:
		available = df[metric].notna().any()
		if not available:
			continue
		plt.figure(figsize=(10, 4))
		sns.barplot(data=df.sort_values(metric, ascending=False), x='run_name', y=metric, color='#55a868')
		plt.xticks(rotation=60, ha='right')
		plt.title(f'{metric.replace("_", " ").title()} by Run')
		plt.tight_layout()
		plt.savefig(PLOTS_DIR / f'cmp_{metric}.png', dpi=150)
		plt.close()

	# Best run summary table saved as CSV
	best_by_acc = df.sort_values('accuracy', ascending=False).head(5)
	best_csv = Path('results/best_runs_top5.csv')
	best_by_acc.to_csv(best_csv, index=False)
	print(f'Saved focused comparisons and best-run summary to {PLOTS_DIR} and {best_csv}')


if __name__ == '__main__':
	main() 