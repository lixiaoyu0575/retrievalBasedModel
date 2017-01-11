import tensorflow as tf
import functools
from tensorflow.contrib.learn.python.learn.metric_spec import MetricSpec

#use the recall@k metric to evaluate our model
def create_evaluation_metrics():
    eval_metrics = {}
    for k in [1, 2, 5, 10]:
        #use functools.partial to convert a function that takes 3 arguments to one that only takes 2 arguments
        eval_metrics["recall_at_%d" % k] = MetricSpec(metric_fn=functools.partial(
            tf.contrib.metrics.streaming_sparse_recall_at_k,#Streaming just means that the metric is accumulated over multiple batches, and sparse refers to the format of our labels
            k=k))
    return eval_metrics
