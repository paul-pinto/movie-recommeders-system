from dataclasses import dataclass


@dataclass
class PredictionResult:
    est: float


class MeanRatingClusteringModel:
    """
    Fallback model compatible with app usage:
    app expects model.predict(user_id, item_id).est
    """

    def __init__(self, user_means, item_means, global_mean):
        self.user_means = user_means
        self.item_means = item_means
        self.global_mean = float(global_mean)

    def predict(self, user_id, item_id):
        user_mean = self.user_means.get(int(user_id), self.global_mean)
        item_mean = self.item_means.get(int(item_id), self.global_mean)
        est = (float(user_mean) + float(item_mean)) / 2.0
        est = max(0.5, min(5.0, est))
        return PredictionResult(est=est)
