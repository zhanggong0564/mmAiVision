from PIL import Image, ImageDraw, ImageFont


OUT_DIR = "docs/assets/yolov5seg-learning"


def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(name, size)


F_TITLE = font(34, True)
F_H = font(22, True)
F = font(18)
F_S = font(15)


COLORS = {
    "bg": "#f7f9fc",
    "ink": "#1f2937",
    "muted": "#64748b",
    "blue": "#2563eb",
    "blue_l": "#dbeafe",
    "green": "#059669",
    "green_l": "#d1fae5",
    "orange": "#ea580c",
    "orange_l": "#ffedd5",
    "purple": "#7c3aed",
    "purple_l": "#ede9fe",
    "red": "#dc2626",
    "red_l": "#fee2e2",
    "gray_l": "#e5e7eb",
}


def rounded(draw, box, fill, outline="#cbd5e1", width=2, radius=16):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def center_text(draw, box, lines, fill=COLORS["ink"], title=False):
    if isinstance(lines, str):
        lines = lines.split("\n")
    fonts = [F_H if title and i == 0 else F for i in range(len(lines))]
    heights = [draw.textbbox((0, 0), t, font=fonts[i])[3] for i, t in enumerate(lines)]
    total = sum(heights) + (len(lines) - 1) * 6
    y = (box[1] + box[3] - total) / 2
    for i, text in enumerate(lines):
        bbox = draw.textbbox((0, 0), text, font=fonts[i])
        x = (box[0] + box[2] - (bbox[2] - bbox[0])) / 2
        draw.text((x, y), text, font=fonts[i], fill=fill)
        y += heights[i] + 6


def arrow(draw, start, end, fill=COLORS["muted"], width=3):
    draw.line([start, end], fill=fill, width=width)
    x1, y1 = start
    x2, y2 = end
    import math

    ang = math.atan2(y2 - y1, x2 - x1)
    size = 11
    pts = [
        (x2, y2),
        (x2 - size * math.cos(ang - 0.45), y2 - size * math.sin(ang - 0.45)),
        (x2 - size * math.cos(ang + 0.45), y2 - size * math.sin(ang + 0.45)),
    ]
    draw.polygon(pts, fill=fill)


def save(img, name):
    img.save(f"{OUT_DIR}/{name}", quality=95)


def model_structure():
    img = Image.new("RGB", (1800, 1050), COLORS["bg"])
    d = ImageDraw.Draw(img)
    d.text((55, 38), "YOLOv5-Seg Model Structure", font=F_TITLE, fill=COLORS["ink"])
    d.text((58, 86), "Backbone -> PAFPN -> Detect branches + Proto branch", font=F, fill=COLORS["muted"])

    inp = (70, 190, 240, 300)
    prep = (300, 190, 500, 300)
    rounded(d, inp, COLORS["gray_l"])
    center_text(d, inp, "Input\n3x640x640")
    rounded(d, prep, COLORS["gray_l"])
    center_text(d, prep, "Preprocess\nRGB / 255")
    arrow(d, (240, 245), (300, 245))

    stages = [
        ((590, 135, 780, 225), COLORS["blue_l"], "Stem\nConv s=2\n320x320"),
        ((590, 255, 780, 345), COLORS["blue_l"], "Stage1\nC3\n160x160"),
        ((590, 375, 780, 465), COLORS["blue_l"], "Stage2\nP3 80x80\nstride 8"),
        ((590, 495, 780, 585), COLORS["blue_l"], "Stage3\nP4 40x40\nstride 16"),
        ((590, 615, 780, 725), COLORS["blue_l"], "Stage4\nP5 20x20\nSPPF"),
    ]
    d.text((612, 103), "Backbone: CSPDarknet", font=F_H, fill=COLORS["blue"])
    arrow(d, (500, 245), (590, 180))
    for i, (box, color, text) in enumerate(stages):
        rounded(d, box, color, outline="#93c5fd")
        center_text(d, box, text)
        if i:
            arrow(d, ((box[0] + box[2]) // 2, stages[i - 1][0][3]), ((box[0] + box[2]) // 2, box[1]))

    neck = (900, 315, 1135, 590)
    d.text((920, 278), "Neck: PAFPN", font=F_H, fill=COLORS["green"])
    rounded(d, neck, COLORS["green_l"], outline="#86efac")
    center_text(d, neck, "Top-down fusion\n+\nBottom-up fusion\n\nOutputs:\nN3 80x80\nN4 40x40\nN5 20x20")
    arrow(d, (780, 420), (900, 385))
    arrow(d, (780, 540), (900, 455))
    arrow(d, (780, 670), (900, 525))

    head = (1260, 260, 1530, 500)
    proto = (1260, 610, 1530, 790)
    rounded(d, head, COLORS["orange_l"], outline="#fdba74")
    center_text(d, head, "Detect branches\n3 anchors each\n\nbbox + obj + cls\n+ mask coeff", title=True)
    rounded(d, proto, COLORS["purple_l"], outline="#c4b5fd")
    center_text(d, proto, "Proto branch\nfrom N3\n\n32 x 160 x 160", title=True)
    arrow(d, (1135, 405), (1260, 370))
    arrow(d, (1135, 405), (1260, 690))

    out1 = (1600, 300, 1740, 405)
    out2 = (1600, 630, 1740, 745)
    rounded(d, out1, "#ffffff")
    center_text(d, out1, "Decode\n+ NMS")
    rounded(d, out2, "#ffffff")
    center_text(d, out2, "coeff @ proto\ncrop by bbox\nbinary mask")
    arrow(d, (1530, 380), (1600, 352))
    arrow(d, (1530, 700), (1600, 690))
    arrow(d, (1530, 380), (1600, 690))

    d.text((70, 910), "Key idea: YOLOv5-Seg predicts boxes and mask coefficients; instance masks are linear combinations of shared Proto masks.", font=F_H, fill=COLORS["ink"])
    save(img, "model-structure.png")


def augmentation_flow():
    img = Image.new("RGB", (1800, 1120), COLORS["bg"])
    d = ImageDraw.Draw(img)
    d.text((55, 38), "Data Pipeline & Augmentations", font=F_TITLE, fill=COLORS["ink"])
    d.text((58, 86), "What each transform changes: image, bbox, polygon/mask", font=F, fill=COLORS["muted"])

    pipeline = [
        ("LoadImage\nFromFile", COLORS["gray_l"]),
        ("LoadLabelme\nAnnotations\nbox_as_mask", COLORS["blue_l"]),
        ("LetterResize\n640x640", COLORS["green_l"]),
        ("PackDetInputs\nfillPoly masks", COLORS["orange_l"]),
    ]
    x = 80
    boxes = []
    for text, color in pipeline:
        box = (x, 170, x + 260, 300)
        boxes.append(box)
        rounded(d, box, color)
        center_text(d, box, text)
        x += 370
    for b1, b2 in zip(boxes, boxes[1:]):
        arrow(d, (b1[2], 235), (b2[0], 235))
    d.text((80, 325), "Current YOLOv5-Seg config uses this stable base pipeline.", font=F, fill=COLORS["muted"])

    rows = [
        ("HSV RandomAug", "Image color only", "bbox: no change", "mask: no change", COLORS["purple_l"]),
        ("RandomFlip", "Flip pixels", "mirror bbox", "mirror polygon points", COLORS["green_l"]),
        ("RandomAffine", "warp image", "transform bbox", "transform polygon; mask later", COLORS["orange_l"]),
        ("Mosaic", "4 images on 2S canvas", "scale + shift + clip", "scale + shift + clip polygons", COLORS["blue_l"]),
        ("MixUp", "linear blend pixels", "merge bboxes", "merge polygons", COLORS["red_l"]),
    ]
    y = 440
    d.text((80, 390), "Implemented optional augmentations", font=F_H, fill=COLORS["ink"])
    for name, image_change, bbox_change, mask_change, color in rows:
        box = (80, y, 1720, y + 95)
        rounded(d, box, "#ffffff", outline="#d1d5db", radius=12)
        d.rectangle((80, y, 350, y + 95), fill=color)
        d.text((105, y + 18), name, font=F_H, fill=COLORS["ink"])
        d.text((390, y + 18), image_change, font=F, fill=COLORS["ink"])
        d.text((820, y + 18), bbox_change, font=F, fill=COLORS["ink"])
        d.text((1190, y + 18), mask_change, font=F, fill=COLORS["ink"])
        y += 115

    d.text((95, 1020), "Design choice: geometric transforms update polygons first; dense masks are rasterized at the very end by PackDetInputs.", font=F_H, fill=COLORS["ink"])
    save(img, "data-augmentations.png")


def training_flow():
    img = Image.new("RGB", (1800, 1050), COLORS["bg"])
    d = ImageDraw.Draw(img)
    d.text((55, 38), "YOLOv5-Seg Training Flow", font=F_TITLE, fill=COLORS["ink"])
    d.text((58, 86), "From LabelMe annotations to four loss terms", font=F, fill=COLORS["muted"])

    items = [
        ((80, 170, 330, 300), COLORS["blue_l"], "Dataset\nJSON -> instances"),
        ((440, 170, 700, 300), COLORS["green_l"], "Transforms\nbbox / polygon / masks"),
        ((810, 170, 1050, 300), COLORS["gray_l"], "Preprocess\nnormalize + batch"),
        ((1160, 170, 1440, 300), COLORS["orange_l"], "Backbone + Neck\nN3 / N4 / N5"),
        ((1520, 170, 1730, 300), COLORS["purple_l"], "SegHead\npred + proto"),
    ]
    for box, color, text in items:
        rounded(d, box, color)
        center_text(d, box, text)
    for a, b in zip(items, items[1:]):
        arrow(d, (a[0][2], 235), (b[0][0], 235))

    assigner = (220, 470, 560, 650)
    detloss = (720, 430, 1070, 600)
    maskloss = (720, 670, 1070, 840)
    total = (1260, 545, 1600, 725)
    rounded(d, assigner, COLORS["blue_l"], outline="#93c5fd")
    center_text(d, assigner, "YOLOv5BatchAssigner\n\nanchor ratio match\n+ neighbor grids\n+ gt_idx")
    rounded(d, detloss, COLORS["orange_l"], outline="#fdba74")
    center_text(d, detloss, "Detection losses\n\nloss_bbox = CIoU\nloss_obj = BCE\nloss_cls = BCE")
    rounded(d, maskloss, COLORS["purple_l"], outline="#c4b5fd")
    center_text(d, maskloss, "Mask loss\n\ncoeff @ proto\nBCE inside bbox\narea normalized")
    rounded(d, total, COLORS["green_l"], outline="#86efac")
    center_text(d, total, "Total loss\n\nbbox + obj + cls + mask")

    arrow(d, (1625, 300), (390, 470))
    arrow(d, (560, 560), (720, 515))
    arrow(d, (560, 560), (720, 755))
    arrow(d, (1070, 515), (1260, 610))
    arrow(d, (1070, 755), (1260, 660))

    d.text((90, 930), "Evaluation uses LabelmeSegMetric: compute mask IoU per image, store only score/label/TP/GT counts, then calculate mAP.", font=F_H, fill=COLORS["ink"])
    save(img, "training-flow.png")


if __name__ == "__main__":
    model_structure()
    augmentation_flow()
    training_flow()
