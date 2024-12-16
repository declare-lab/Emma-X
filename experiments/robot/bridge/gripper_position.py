import matplotlib
import numpy as np
import torch
from transformers import SamModel, SamProcessor, pipeline


checkpoint = "google/owlvit-base-patch16"
detector = pipeline(model=checkpoint, task="zero-shot-object-detection", device="cuda")
sam_model = SamModel.from_pretrained("facebook/sam-vit-base").cuda()
sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

image_dims = (256, 256)


def get_bounding_boxes(img, prompt="the black robotic gripper"):
    predictions = detector(img, candidate_labels=[prompt], threshold=0.01)

    return predictions


def show_box(box, ax, meta, color):
    x0, y0 = box["xmin"], box["ymin"]
    w, h = box["xmax"] - box["xmin"], box["ymax"] - box["ymin"]
    ax.add_patch(
        matplotlib.patches.FancyBboxPatch((x0, y0), w, h, edgecolor=color, facecolor=(0, 0, 0, 0), lw=2, label="hehe")
    )
    ax.text(x0, y0 + 10, "{:.3f}".format(meta["score"]), color="white")


def get_median(mask, p):
    row_sum = np.sum(mask, axis=1)
    cumulative_sum = np.cumsum(row_sum)

    if p >= 1.0:
        p = 1

    total_sum = np.sum(row_sum)
    threshold = p * total_sum

    return np.argmax(cumulative_sum >= threshold)


def get_gripper_mask(img, pred):
    box = [
        round(pred["box"]["xmin"], 2),
        round(pred["box"]["ymin"], 2),
        round(pred["box"]["xmax"], 2),
        round(pred["box"]["ymax"], 2),
    ]

    inputs = sam_processor(img, input_boxes=[[[box]]], return_tensors="pt")

    for k in inputs.keys():
        inputs[k] = inputs[k].cuda()
    with torch.no_grad():
        outputs = sam_model(**inputs)

    mask = (
        sam_processor.image_processor.post_process_masks(
            outputs.pred_masks, inputs["original_sizes"], inputs["reshaped_input_sizes"]
        )[0][0][0]
        .cpu()
        .numpy()
    )

    return mask


def sq(w, h):
    return np.concatenate(
        [
            (np.arange(w * h).reshape(h, w) % w)[:, :, None],
            (np.arange(w * h).reshape(h, w) // w)[:, :, None],
        ],
        axis=-1,
    )


def mask_to_pos_weighted(mask):
    pos = sq(*image_dims)

    weight = pos[:, :, 0] + pos[:, :, 1]
    weight = weight * weight

    x = np.sum(mask * pos[:, :, 0] * weight) / np.sum(mask * weight)
    y = get_median(mask * weight, 0.95)

    return x, y


def mask_to_pos_naive(mask):
    pos = sq(*image_dims)
    weight = pos[:, :, 0] + pos[:, :, 1]
    min_pos = np.argmax((weight * mask).flatten())

    return min_pos % image_dims[0] - (image_dims[0] / 16), min_pos // image_dims[0] - (image_dims[0] / 24)


def get_gripper_pos_raw(img):
    # img = Image.fromarray(img.numpy())
    predictions = get_bounding_boxes(img)

    if len(predictions) > 0:
        mask = get_gripper_mask(img, predictions[0])
        pos = mask_to_pos_naive(mask)
    else:
        mask = np.zeros(image_dims)
        pos = (-1, -1)
        predictions = [None]

    # return (int(pos[0]), int(pos[1])), mask, predictions[0]
    return (int(pos[0]*224/256), int(pos[1]*224/256)), mask, predictions[0]


if __name__ == "__main__":
    pass
