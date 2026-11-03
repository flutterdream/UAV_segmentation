CLASS_NAMES = (
    "Background",
    "Building",
    "Road",
    "Water",
    "Barren",
    "Vegetation",
    "Agricultural",
    "Vehicle",
)

NUM_CLASSES = len(CLASS_NAMES)
IGNORE_RAW_ID = 0
IGNORE_TRAIN_ID = 255


def raw_to_train(mask):
    """Map competition IDs 1..8 to train IDs 0..7 and raw ignore 0 to 255."""
    result = mask.copy()
    result[mask == IGNORE_RAW_ID] = IGNORE_TRAIN_ID
    valid = mask > IGNORE_RAW_ID
    result[valid] = mask[valid] - 1
    return result


def train_to_raw(mask):
    """Map train IDs 0..7 back to competition IDs 1..8."""
    return mask + 1

