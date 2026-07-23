from data_utils import normalize_sentiment


def test_normalize_sentiment_maps_common_labels() -> None:
    assert normalize_sentiment("Extreme Fear") == "Fear"
    assert normalize_sentiment("Extreme Greed") == "Greed"
    assert normalize_sentiment("Neutral") == "Neutral"
    assert normalize_sentiment("Unknown") == "Neutral"
