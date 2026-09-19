"""
Train and evaluate MLlib classifiers on Spark, then save the
best-performing pipeline model.

Usage:
    python src/train.py [--config config.yaml]

Loads the data into a Spark DataFrame, builds an MLlib Pipeline (indexing +
one-hot encoding + assembling + scaling + classifier) for each configured
model, evaluates both with Spark's BinaryClassificationEvaluator, and
persists the higher-AUC PipelineModel to disk using Spark's native model
format.

"""

import argparse
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(__file__))
from spark_utils import load_config, get_spark_session
from pipeline import build_pipeline

from pyspark.sql.functions import col
from pyspark.sql import functions as F
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.functions import vector_to_array


def load_data(spark, config):
    df = spark.read.csv(config["data_path"], header=True, inferSchema=True)
    df = df.withColumn("label", col(config["target_col"]).cast("double"))
    return df


def evaluate_predictions(predictions):
    """Compute AUC (via Spark's evaluator) plus accuracy/precision/recall/F1
    from a small collected confusion matrix — fine to collect since this is
    a 2x2 table, regardless of dataset size.
    """
    evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    auc = evaluator.evaluate(predictions)

    counts = (
        predictions.groupBy("label", "prediction").count().collect()
    )
    confusion = {(int(r["label"]), int(r["prediction"])): r["count"] for r in counts}
    tp = confusion.get((1, 1), 0)
    tn = confusion.get((0, 0), 0)
    fp = confusion.get((0, 1), 0)
    fn = confusion.get((1, 0), 0)

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "auc": auc, "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def get_roc_points(predictions):
    """
    Extract (label, positive-class-probability) pairs and compute an ROC
    curve with numpy. Collecting to the driver is reasonable here since
    it's just the test split of a modest dataset — for a genuinely huge
    test set you'd sample first.
    """
    prob_df = predictions.select(
        col("label"),
        vector_to_array(col("probability"))[1].alias("prob_positive"),
    )
    pdf = prob_df.toPandas()

    labels = pdf["label"].to_numpy()
    scores = pdf["prob_positive"].to_numpy()

    order = np.argsort(-scores)
    labels_sorted = labels[order]

    tp_cum = np.cumsum(labels_sorted == 1)
    fp_cum = np.cumsum(labels_sorted == 0)
    n_pos = np.sum(labels == 1)
    n_neg = np.sum(labels == 0)

    tpr = tp_cum / n_pos if n_pos else np.zeros_like(tp_cum, dtype=float)
    fpr = fp_cum / n_neg if n_neg else np.zeros_like(fp_cum, dtype=float)
    return np.concatenate([[0], fpr]), np.concatenate([[0], tpr])


def main():
    parser = argparse.ArgumentParser(description="Train MLlib churn classifiers.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    spark = get_spark_session(config)

    numeric_features = config["numeric_features"]
    categorical_features = config["categorical_features"]

    print(f"Loading data from {config['data_path']} ...")
    df = load_data(spark, config)
    n_total = df.count()
    churn_rate = df.selectExpr("avg(label) as rate").collect()[0]["rate"]
    print(f"{n_total} rows loaded. Churn rate: {churn_rate*100:.1f}%\n")

    train_df, test_df = df.randomSplit(
        [1 - config.get("test_size", 0.2), config.get("test_size", 0.2)],
        seed=config.get("random_state", 42),
    )
    train_df.cache()
    test_df.cache()
    print(f"Train rows: {train_df.count()}  |  Test rows: {test_df.count()}\n")

    # Balanced class weights (mirrors sklearn's class_weight="balanced"):
    # weight = n_total / (n_classes * n_in_this_class), computed from the
    # training split only, so the model doesn't just learn to always
    # predict "no churn" on this ~22%-positive dataset.
    class_counts = {row["label"]: row["count"] for row in train_df.groupBy("label").count().collect()}
    n_train = sum(class_counts.values())
    n_classes = len(class_counts)
    weight_map = {label: n_train / (n_classes * count) for label, count in class_counts.items()}
    print(f"Class weights (balanced): {weight_map}\n")

    weight_expr = None
    for label_val, w in weight_map.items():
        cond = (col("label") == label_val)
        weight_expr = F.when(cond, w) if weight_expr is None else weight_expr.when(cond, w)
    train_df = train_df.withColumn("class_weight", weight_expr)

    results = {}
    fitted_models = {}
    roc_points = {}

    for model_key in config.get("models", ["logistic_regression", "random_forest"]):
        display_name = model_key.replace("_", " ").title()
        print(f"Training {display_name}...")
        pipeline = build_pipeline(model_key, numeric_features, categorical_features, config.get("random_state", 42))
        fitted = pipeline.fit(train_df)
        predictions = fitted.transform(test_df)

        metrics = evaluate_predictions(predictions)
        results[display_name] = metrics
        fitted_models[display_name] = fitted
        roc_points[display_name] = get_roc_points(predictions)

        print(f"  AUC:       {metrics['auc']:.3f}")
        print(f"  Accuracy:  {metrics['accuracy']:.3f}")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall:    {metrics['recall']:.3f}")
        print(f"  F1:        {metrics['f1']:.3f}\n")

    best_name = max(results, key=lambda k: results[k]["auc"])
    best_model = fitted_models[best_name]
    print(f"Best model by AUC: {best_name} ({results[best_name]['auc']:.3f})")

    model_path = config["model_path"]
    best_model.write().overwrite().save(model_path)
    print(f"Saved best pipeline model to {model_path}/ (Spark native model format)")

    # --- ROC plot ---
    os.makedirs(config["figures_dir"], exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, (fpr, tpr) in roc_points.items():
        ax.plot(fpr, tpr, label=f"{name} (AUC = {results[name]['auc']:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (MLlib models)")
    ax.legend()
    fig.tight_layout()
    roc_path = os.path.join(config["figures_dir"], "roc_curves.png")
    fig.savefig(roc_path, dpi=150)
    plt.close(fig)
    print(f"ROC curve saved to {roc_path}")

    # --- Metrics report ---
    lines = ["# Churn MLlib Pipeline — Training Report", ""]
    lines.append(f"Rows: {n_total}  |  Train: {train_df.count()}  |  Test: {test_df.count()}  |  Churn rate: {churn_rate*100:.1f}%")
    lines.append("")
    lines.append("| Model | AUC | Accuracy | Precision | Recall | F1 |")
    lines.append("|---|---|---|---|---|---|")
    for name, m in results.items():
        marker = " **(best)**" if name == best_name else ""
        lines.append(f"| {name}{marker} | {m['auc']:.3f} | {m['accuracy']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} |")
    lines.append("")
    lines.append("## Confusion matrices")
    lines.append("")
    for name, m in results.items():
        c = m["confusion"]
        lines.append(f"**{name}**: TP={c['tp']}, TN={c['tn']}, FP={c['fp']}, FN={c['fn']}")
        lines.append("")
    lines.append("## ROC Curves")
    lines.append("![ROC Curves](figures/roc_curves.png)")

    with open(config["metrics_report_path"], "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Metrics report saved to {config['metrics_report_path']}")

    spark.stop()


if __name__ == "__main__":
    main()