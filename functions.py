def classify_confidence(conf):
    if conf <= 0.5:
        return "Faulty"
    elif conf <= 0.8:
        return "Middle"
    return "Good"


def assign_status_to_detections(detections):
    statuses = [classify_confidence(d["confidence"]) for d in detections]

    if "Faulty" in statuses:
        return "Faulty"
    elif "Middle" in statuses:
        return "Middle"
    return "Good"
