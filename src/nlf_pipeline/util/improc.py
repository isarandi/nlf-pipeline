import numpy as np
import cv2
from sklearn import linear_model


class Clahe:
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(12, 12))
        lut = np.arange(256)
        lut[:24] = np.linspace(0, 72, 24)
        lut[24:] = np.linspace(72, 255, 256 - 24)
        lut = np.uint8(lut)
        self.lut = lut

    def __call__(self, frame):
        factor = 256 if frame.dtype == np.uint16 else 1
        is_valid_mask = np.any(frame > 20 * factor, axis=-1)
        maxval = 255 if frame.dtype == np.uint8 else 65535
        recolor_border(frame, is_valid_mask, border_value=(maxval, maxval, maxval))

        cv2.cvtColor(frame, cv2.COLOR_RGB2YUV, dst=frame)
        y = self.clahe.apply(frame[..., 0])
        if frame.dtype == np.uint16:
            frame_uint8 = (frame // 256).astype(np.uint8)
            y = (y // 256).astype(np.uint8)
            cv2.LUT(y, self.lut, dst=y)
            frame_uint8[..., 0] = y
            cv2.cvtColor(frame_uint8, cv2.COLOR_YUV2RGB, dst=frame_uint8)
            return frame_uint8
        else:
            cv2.LUT(y, self.lut, dst=y)
            frame[..., 0] = y
            cv2.cvtColor(frame, cv2.COLOR_YUV2RGB, dst=frame)
            return frame


#
# def clahe(frame):
#     recolor_border(frame, border_value=(255, 255, 255))
#     clahe = cv2.createCLAHE(clipLimit=2., tileGridSize=(12, 12))
#     yuv = cv2.cvtColor(frame, cv2.COLOR_RGB2YUV)
#     y, u, v = cv2.split(yuv)
#     y = clahe.apply(y)
#
#     lut = np.arange(256)
#     lut[:24] = np.linspace(0, 72, 24)
#     lut[24:] = np.linspace(72, 255, 256 - 24)
#     lut = np.uint8(lut)
#     y = cv2.LUT(y, lut)
#     yuv = cv2.merge((y, u, v))
#     return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)


def recolor_border(im, is_valid_mask, border_value=(127, 127, 127)):
    h, w = im.shape[:2]
    im_changed = im

    # bottom:
    last_valid_index_per_col = h - np.argmax(is_valid_mask[::-1], axis=0)
    is_any_valid_per_col = np.any(is_valid_mask, axis=0)
    last_valid_index_per_col[~is_any_valid_per_col] = 0

    col_inds = np.arange(w)
    quantile = 1e-1
    ransac_start = linear_model.QuantileRegressor(quantile=quantile, alpha=0, solver='highs')
    ransac_end = linear_model.QuantileRegressor(quantile=1 - quantile, alpha=0, solver='highs')
    fitted = ransac_end.fit(col_inds[:, np.newaxis], last_valid_index_per_col)
    offset = fitted.intercept_
    if offset < h - 1:
        offset -= 1

    slope = fitted.coef_[0]
    y1 = offset
    y2 = offset + slope * w
    y3 = max(h, y1)
    y4 = max(h, y2)
    points = np.array([[0, y1], [w, y2], [w, y3], [0, y4]], np.int32)
    im_changed = cv2.fillPoly(im_changed, [points], border_value, lineType=cv2.LINE_AA)

    # top:
    first_valid_index_per_col = np.argmax(is_valid_mask, axis=0)
    first_valid_index_per_col[~is_any_valid_per_col] = h
    fitted = ransac_start.fit(col_inds[:, np.newaxis], first_valid_index_per_col)
    offset = fitted.intercept_
    if offset > 0:
        offset += 1

    slope = fitted.coef_[0]
    y1 = offset
    y2 = offset + slope * w
    y3 = min(0, y1)
    y4 = min(0, y2)
    points = np.array([[0, y1], [w, y2], [w, y3], [0, y4]], np.int32)
    im_changed = cv2.fillPoly(im_changed, [points], border_value, lineType=cv2.LINE_AA)

    # left:
    first_valid_index_per_row = np.argmax(is_valid_mask, axis=1)
    is_any_valid_per_row = np.any(is_valid_mask, axis=1)
    first_valid_index_per_row[~is_any_valid_per_row] = w
    row_inds = np.arange(h)
    fitted = ransac_start.fit(row_inds[:, np.newaxis], first_valid_index_per_row)
    offset = fitted.intercept_
    if offset > 0:
        offset += 1
    slope = fitted.coef_[0]
    x1 = offset
    x2 = offset + slope * h
    x3 = min(0, x1)
    x4 = min(0, x2)
    points = np.array([[x1, 0], [x2, h], [x3, h], [x4, 0]], np.int32)
    im_changed = cv2.fillPoly(im_changed, [points], border_value, lineType=cv2.LINE_AA)

    # right:
    last_valid_index_per_row = w - np.argmax(is_valid_mask[:, ::-1], axis=1)
    last_valid_index_per_row[~is_any_valid_per_row] = 0
    fitted = ransac_end.fit(row_inds[:, np.newaxis], last_valid_index_per_row)
    offset = fitted.intercept_
    if offset < w - 1:
        offset -= 1
    slope = fitted.coef_[0]
    x1 = offset
    x2 = offset + slope * h
    x3 = max(w, x1)
    x4 = max(w, x2)
    points = np.array([[x1, 0], [x2, h], [x3, h], [x4, 0]], np.int32)
    im_changed = cv2.fillPoly(im_changed, [points], border_value, lineType=cv2.LINE_AA)
    return im_changed
