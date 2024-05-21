import argparse

from scidocs.paths import DataPaths
from scidocs import get_scidocs_metrics

data_paths = DataPaths()

parser = argparse.ArgumentParser()
parser.add_argument("class_path", default="mag_mesh_embed.jsonl")
parser.add_argument("user_path", default="view_cite_read_embed.jsonl")
parser.add_argument("recomm_path", default="recomm_embed.jsonl")

args = parser.parse_args()

classification_embeddings_path = args.class_path
user_activity_and_citations_embeddings_path = args.user_path
recomm_embeddings_path = args.recomm_path

# run the evaluation
scidocs_metrics = get_scidocs_metrics(
    data_paths,
    classification_embeddings_path,
    user_activity_and_citations_embeddings_path,
    recomm_embeddings_path,
    val_or_test="test",  # set to 'val' if tuning hyperparams
    n_jobs=4,  # the classification tasks can be parallelized
    cuda_device=-1,  # the recomm task can use a GPU if this is set to 0, 1, etc
)

print(scidocs_metrics)
