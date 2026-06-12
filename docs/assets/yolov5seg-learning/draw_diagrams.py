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


def letterresize_detail():
    img = Image.new("RGB", (1800, 1050), COLORS["bg"])
    d = ImageDraw.Draw(img)
    d.text((55, 38), "LetterResize: keep aspect ratio + padding", font=F_TITLE, fill=COLORS["ink"])
    d.text((58, 86), "Image, bboxes and polygons are transformed by the same scale and offset.", font=F, fill=COLORS["muted"])

    orig = (110, 250, 560, 610)
    resized = (720, 300, 1060, 555)
    final = (1240, 210, 1640, 610)
    rounded(d, orig, "#ffffff", outline="#94a3b8", radius=10)
    d.text((230, 210), "Original image H x W", font=F_H, fill=COLORS["ink"])
    d.rectangle((210, 335, 420, 505), outline=COLORS["red"], width=5)
    d.polygon([(250, 380), (370, 360), (420, 470), (285, 500)], outline=COLORS["purple"], fill=None)
    d.line([(250, 380), (370, 360), (420, 470), (285, 500), (250, 380)], fill=COLORS["purple"], width=5)

    rounded(d, resized, COLORS["green_l"], outline="#86efac", radius=10)
    d.text((790, 260), "Resize by r", font=F_H, fill=COLORS["green"])
    d.rectangle((790, 365, 950, 495), outline=COLORS["red"], width=5)
    d.line([(820, 400), (920, 385), (955, 470), (850, 495), (820, 400)], fill=COLORS["purple"], width=5)

    rounded(d, final, "#ffffff", outline="#94a3b8", radius=10)
    d.rectangle(final, outline="#64748b", width=4)
    d.rectangle((1240, 210, 1640, 285), fill="#e2e8f0")
    d.rectangle((1240, 535, 1640, 610), fill="#e2e8f0")
    d.text((1302, 170), "Final 640 x 640", font=F_H, fill=COLORS["ink"])
    d.rectangle((1340, 365, 1500, 495), outline=COLORS["red"], width=5)
    d.line([(1370, 400), (1470, 385), (1505, 470), (1400, 495), (1370, 400)], fill=COLORS["purple"], width=5)

    arrow(d, (560, 430), (720, 430), fill=COLORS["green"])
    arrow(d, (1060, 430), (1240, 430), fill=COLORS["green"])
    d.text((610, 390), "scale", font=F_H, fill=COLORS["green"])
    d.text((1110, 390), "pad", font=F_H, fill=COLORS["green"])

    formula = (140, 760, 1660, 940)
    rounded(d, formula, COLORS["blue_l"], outline="#93c5fd", radius=14)
    d.text((185, 790), "Coordinate rule", font=F_H, fill=COLORS["blue"])
    d.text((185, 835), "r = min(640 / H, 640 / W)", font=F, fill=COLORS["ink"])
    d.text((640, 835), "x' = x * r + left", font=F, fill=COLORS["ink"])
    d.text((1020, 835), "y' = y * r + top", font=F, fill=COLORS["ink"])
    d.text((185, 885), "Saved for restore: scale_factor=(r,r), pad_param=[top,bottom,left,right]", font=F, fill=COLORS["ink"])
    save(img, "augmentation-letterresize.png")


def flip_hsv_detail():
    img = Image.new("RGB", (1800, 1000), COLORS["bg"])
    d = ImageDraw.Draw(img)
    d.text((55, 38), "RandomFlip & HSV Augmentation", font=F_TITLE, fill=COLORS["ink"])
    d.text((58, 86), "Flip changes geometry; HSV changes color only.", font=F, fill=COLORS["muted"])

    left = (100, 210, 620, 620)
    right = (760, 210, 1280, 620)
    rounded(d, left, "#ffffff", outline="#94a3b8", radius=12)
    rounded(d, right, "#ffffff", outline="#94a3b8", radius=12)
    d.text((255, 168), "Before flip", font=F_H, fill=COLORS["ink"])
    d.text((930, 168), "Horizontal flip", font=F_H, fill=COLORS["ink"])
    d.rectangle((210, 330, 420, 510), outline=COLORS["red"], width=5)
    d.line([(250, 385), (395, 350), (430, 500), (275, 520), (250, 385)], fill=COLORS["purple"], width=5)
    d.rectangle((960, 330, 1170, 510), outline=COLORS["red"], width=5)
    d.line([(1130, 385), (985, 350), (950, 500), (1105, 520), (1130, 385)], fill=COLORS["purple"], width=5)
    arrow(d, (620, 415), (760, 415), fill=COLORS["green"])
    d.text((645, 370), "x' = W - x", font=F, fill=COLORS["green"])

    panel = (1380, 210, 1680, 620)
    rounded(d, panel, COLORS["purple_l"], outline="#c4b5fd", radius=14)
    center_text(d, panel, "HSV RandomAug\n\nBGR -> HSV\nLUT for H/S/V\nHSV -> BGR\n\nbbox unchanged\npolygon unchanged", title=True)

    rule = (140, 740, 1660, 900)
    rounded(d, rule, COLORS["green_l"], outline="#86efac", radius=14)
    d.text((185, 770), "Flip coordinate rule", font=F_H, fill=COLORS["green"])
    d.text((185, 815), "Horizontal bbox: x1_new = W - x2, x2_new = W - x1", font=F, fill=COLORS["ink"])
    d.text((850, 815), "Vertical bbox: y1_new = H - y2, y2_new = H - y1", font=F, fill=COLORS["ink"])
    d.text((185, 860), "Polygon points use the same mirror rule point by point.", font=F, fill=COLORS["ink"])
    save(img, "augmentation-flip-hsv.png")


def affine_detail():
    img = Image.new("RGB", (1800, 1050), COLORS["bg"])
    d = ImageDraw.Draw(img)
    d.text((55, 38), "RandomAffine: transform image and polygon together", font=F_TITLE, fill=COLORS["ink"])
    d.text((58, 86), "Translation, scale, rotation and shear are represented by one affine matrix.", font=F, fill=COLORS["muted"])

    steps = [
        ((90, 230, 390, 470), COLORS["gray_l"], "Input\nimage + bbox\n+ polygon"),
        ((520, 230, 820, 470), COLORS["blue_l"], "Build matrix M\ncenter -> rotate/scale\n-> shear -> translate"),
        ((950, 170, 1250, 360), COLORS["green_l"], "Image\ncv2.warpAffine"),
        ((950, 450, 1250, 640), COLORS["purple_l"], "Polygon points\npts @ M"),
        ((1370, 310, 1670, 540), COLORS["orange_l"], "Recompute bbox\nclip to canvas\nfilter bad boxes"),
    ]
    for box, color, text in steps:
        rounded(d, box, color)
        center_text(d, box, text)
    arrow(d, (390, 350), (520, 350), fill=COLORS["blue"])
    arrow(d, (820, 350), (950, 265), fill=COLORS["green"])
    arrow(d, (820, 350), (950, 545), fill=COLORS["purple"])
    arrow(d, (1250, 265), (1370, 390), fill=COLORS["orange"])
    arrow(d, (1250, 545), (1370, 460), fill=COLORS["orange"])

    # mini before/after sketches
    d.rectangle((150, 520, 330, 670), outline=COLORS["red"], width=5)
    d.line([(180, 560), (300, 545), (325, 635), (205, 660), (180, 560)], fill=COLORS["purple"], width=5)
    d.polygon([(1010, 705), (1190, 660), (1235, 790), (1055, 840)], outline=COLORS["red"], fill=None)
    d.line([(1050, 725), (1165, 690), (1210, 785), (1090, 825), (1050, 725)], fill=COLORS["purple"], width=5)

    note = (140, 875, 1660, 985)
    rounded(d, note, "#ffffff", outline="#cbd5e1", radius=14)
    d.text((180, 900), "Important design", font=F_H, fill=COLORS["ink"])
    d.text((180, 945), "Dense masks are not warped directly. The code transforms polygons first, then PackDetInputs rasterizes masks with fillPoly.", font=F, fill=COLORS["ink"])
    save(img, "augmentation-affine.png")


def mosaic_mixup_detail():
    img = Image.new("RGB", (1800, 1120), COLORS["bg"])
    d = ImageDraw.Draw(img)
    d.text((55, 38), "Mosaic & MixUp", font=F_TITLE, fill=COLORS["ink"])
    d.text((58, 86), "Multi-image augmentations use dataset sampling and merge annotations.", font=F, fill=COLORS["muted"])

    d.text((120, 155), "Mosaic: four images on a 2S x 2S canvas", font=F_H, fill=COLORS["blue"])
    canvas = (150, 220, 670, 740)
    d.rectangle(canvas, fill="#ffffff", outline="#64748b", width=5)
    d.rectangle((150, 220, 410, 480), fill="#dbeafe", outline="#64748b", width=3)
    d.rectangle((410, 220, 670, 480), fill="#d1fae5", outline="#64748b", width=3)
    d.rectangle((150, 480, 410, 740), fill="#ffedd5", outline="#64748b", width=3)
    d.rectangle((410, 480, 670, 740), fill="#ede9fe", outline="#64748b", width=3)
    for txt, pos in [("img0", (245, 335)), ("img1", (505, 335)), ("img2", (245, 595)), ("img3", (505, 595))]:
        d.text(pos, txt, font=F_H, fill=COLORS["ink"])
    d.line([(410, 220), (410, 740)], fill=COLORS["red"], width=4)
    d.line([(150, 480), (670, 480)], fill=COLORS["red"], width=4)
    d.text((340, 760), "random center", font=F, fill=COLORS["red"])

    msteps = [
        ((840, 220, 1120, 330), COLORS["blue_l"], "sample 3 extra images"),
        ((840, 390, 1120, 500), COLORS["green_l"], "resize each image"),
        ((840, 560, 1120, 670), COLORS["orange_l"], "paste quadrants"),
        ((1240, 390, 1540, 500), COLORS["purple_l"], "scale + shift\nbbox/polygon"),
        ((1240, 560, 1540, 670), COLORS["red_l"], "clip to canvas"),
    ]
    for box, color, text in msteps:
        rounded(d, box, color)
        center_text(d, box, text)
    arrow(d, (670, 480), (840, 275), fill=COLORS["blue"])
    arrow(d, (1120, 275), (1120, 390), fill=COLORS["green"])
    arrow(d, (1120, 445), (1120, 560), fill=COLORS["orange"])
    arrow(d, (1120, 615), (1240, 445), fill=COLORS["purple"])
    arrow(d, (1390, 500), (1390, 560), fill=COLORS["red"])

    d.text((120, 855), "MixUp: blend pixels, merge annotations", font=F_H, fill=COLORS["green"])
    a = (150, 910, 390, 1040)
    b = (510, 910, 750, 1040)
    c = (1040, 910, 1360, 1040)
    rounded(d, a, COLORS["blue_l"])
    center_text(d, a, "image A\nannotations A")
    rounded(d, b, COLORS["orange_l"])
    center_text(d, b, "image B\nannotations B")
    rounded(d, c, COLORS["green_l"])
    center_text(d, c, "A*lambda + B*(1-lambda)\nannotations A union B")
    arrow(d, (390, 975), (510, 975), fill=COLORS["muted"])
    arrow(d, (750, 975), (1040, 975), fill=COLORS["green"])
    save(img, "augmentation-mosaic-mixup.png")


if __name__ == "__main__":
    model_structure()
    augmentation_flow()
    training_flow()
    letterresize_detail()
    flip_hsv_detail()
    affine_detail()
    mosaic_mixup_detail()
