"""Builds the MLlib preprocessing + classification pipeline.

Uses pyspark.ml.Pipeline so that every stage (string indexing, one-hot
encoding, feature assembly, scaling, and the classifier itself) is fit
once on the training data and then applied identically — and correctly —
to the test set and to new data at prediction time. This mirrors the
scikit-learn ColumnTransformer + Pipeline pattern, but running on Spark's
distributed DataFrame engine instead of in-memory numpy arrays.
"""

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier

CLASSIFIER_BUILDERS = {
    "logistic_regression": lambda seed: LogisticRegression(
        featuresCol="scaled_features", labelCol="label", maxIter=50,
    ),
    "random_forest": lambda seed: RandomForestClassifier(
        featuresCol="scaled_features", labelCol="label", numTrees=200, seed=seed,
    ),
}


def build_pipeline(classifier_key, numeric_features, categorical_features, random_state=42):
    stages = []

    indexed_cols = []
    for col in categorical_features:
        indexer = StringIndexer(inputCol=col, outputCol=f"{col}_idx", handleInvalid="keep")
        stages.append(indexer)
        indexed_cols.append(f"{col}_idx")

    encoded_cols = []
    for col in indexed_cols:
        encoder = OneHotEncoder(inputCol=col, outputCol=f"{col}_ohe")
        stages.append(encoder)
        encoded_cols.append(f"{col}_ohe")

    assembler = VectorAssembler(
        inputCols=list(numeric_features) + encoded_cols,
        outputCol="raw_features",
    )
    stages.append(assembler)

    scaler = StandardScaler(inputCol="raw_features", outputCol="scaled_features", withMean=True, withStd=True)
    stages.append(scaler)

    classifier = CLASSIFIER_BUILDERS[classifier_key](random_state)
    stages.append(classifier)

    return Pipeline(stages=stages)