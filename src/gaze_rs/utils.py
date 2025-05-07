
def center_crop(source, target):
    """Crop `source` tensor to the shape of `target` tensor (center crop)."""
    src_h, src_w = source.shape[2], source.shape[3]
    tgt_h, tgt_w = target.shape[2], target.shape[3]

    offset_h = (src_h - tgt_h) // 2
    offset_w = (src_w - tgt_w) // 2

    return source[:, :, offset_h:offset_h + tgt_h, offset_w:offset_w + tgt_w]

