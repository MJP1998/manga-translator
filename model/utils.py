from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.engine.base_layer import Layer
from keras import backend as K
from tensorflow.keras import models
import numpy as np
import cv2


class Mish(Layer):
    '''
    Mish Activation Function.
    .. math::
        mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))
    Shape:
        - Input: Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
        - Output: Same shape as the input.
    Examples:
        >>> X_input = Input(input_shape)
        >>> X = Mish()(X_input)
    '''

    def __init__(self, **kwargs):
        super(Mish, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return inputs * K.tanh(K.softplus(inputs))

    def get_config(self):
        base_config = super(Mish, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

def load_model(path='./model/yolov4.h5'):
    return models.load_model(path, custom_objects={'Mish': Mish}, compile=False)


def yolo_head(feats, anchors, num_classes, input_shape):
    """from https://github.com/Ma-Dan/keras-yolo4 """
    num_anchors = len(anchors)
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

    grid_shape = K.shape(feats)[1:3] # height, width
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]), [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]), [grid_shape[0], 1, 1, 1])
    grid = K.concatenate([grid_x, grid_y])
    grid = K.cast(grid, K.dtype(feats))

    feats = K.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])
    box_x_y = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[..., ::-1], K.dtype(feats))
    box_w_h = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[..., ::-1], K.dtype(feats))
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.sigmoid(feats[..., 5:])

    return box_x_y, box_w_h, box_confidence, box_class_probs

def yolo_correct_boxes(box_xy, box_wh, image_shape):
    '''from https://github.com/Ma-Dan/keras-yolo4
        Get corrected boxes'''
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    image_shape = K.cast(image_shape, K.dtype(box_yx))

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes =  K.concatenate([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ])

    # Scale boxes back to original image shape.
    boxes *= K.concatenate([image_shape, image_shape])
    return boxes


def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    '''from https://github.com/Ma-Dan/keras-yolo4
    Process Conv layer output'''
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats, anchors, num_classes, input_shape)
    boxes = yolo_correct_boxes(box_xy, box_wh, image_shape)
    boxes = K.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs
    box_scores = K.reshape(box_scores, [-1, num_classes])
    return boxes, box_scores


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _process_feats(out, anchors, mask, input_shape):
    grid_h, grid_w, num_boxes = map(int, out.shape[1: 4])

    anchors = [anchors[i] for i in mask]
    anchors_tensor = np.array(anchors).reshape(1, 1, len(anchors), 2)

    # Reshape to batch, height, width, num_anchors, box_params.
    out = out[0]
    box_xy = _sigmoid(out[..., :2])
    box_wh = np.exp(out[..., 2:4])
    box_wh = box_wh * anchors_tensor

    box_confidence = _sigmoid(out[..., 4])
    box_confidence = np.expand_dims(box_confidence, axis=-1)
    box_class_probs = _sigmoid(out[..., 5:])

    col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w)
    row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h)

    col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    grid = np.concatenate((col, row), axis=-1)

    box_xy += grid
    box_xy /= (grid_w, grid_h)
    box_wh /= input_shape
    box_xy -= (box_wh / 2.)  # 坐标格式是左上角xy加矩形宽高wh，xywh都除以图片边长归一化了。
    boxes = np.concatenate((box_xy, box_wh), axis=-1)

    return boxes, box_confidence, box_class_probs

def _nms_boxes(boxes, scores):
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 1)
        h1 = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= 0.6)[0]
        order = order[inds + 1]

    keep = np.array(keep)

    return keep


def _filter_boxes(boxes, box_confidences, box_class_probs):
    box_scores = box_confidences * box_class_probs
    box_classes = np.argmax(box_scores, axis=-1)
    box_class_scores = np.max(box_scores, axis=-1)
    print(box_class_scores)
    pos = np.where(box_class_scores >= 0.)

    boxes = boxes[pos]
    classes = box_classes[pos]
    scores = box_class_scores[pos]

    return boxes, classes, scores


def _yolo_out(outs, shape, input_shape):
    masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    anchors = [[12, 16], [19, 36], [40, 28], [36, 75], [76, 55],
               [72, 146], [142, 110], [192, 243], [459, 401]]

    boxes, classes, scores = [], [], []

    for out, mask in zip(outs, masks):
        b, c, s = _process_feats(out, anchors, mask, input_shape)
        b, c, s = _filter_boxes(b, c, s)
        boxes.append(b)
        classes.append(c)
        scores.append(s)
    boxes = np.concatenate(boxes)
    classes = np.concatenate(classes)
    scores = np.concatenate(scores)

    # boxes坐标格式是左上角xy加矩形宽高wh，xywh都除以图片边长归一化了。
    # Scale boxes back to original image shape.
    w, h = shape[1], shape[0]
    image_dims = [w, h, w, h]
    boxes = boxes * image_dims

    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]

        keep = _nms_boxes(b, s)

        nboxes.append(b[keep])
        nclasses.append(c[keep])
        nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    # 换坐标
    boxes[:, [2, 3]] = boxes[:, [0, 1]] + boxes[:, [2, 3]]

    return boxes, scores, classes

def detect(img, model):
    image_shape = img.shape
    outs = model.predict(np.array([cv2.resize(img, (416, 416))]))
    anchors = [[10,14],  [23,27],  [37,58],  [81,82],  [135,169],  [344,319]]
    masks = [[1, 2, 3]]
    a1 = np.reshape(outs[2], (1, 416 // 32, 416 // 32, 3, 5 + 1))
    a2 = np.reshape(outs[1], (1, 416 // 16, 416 // 16, 3, 5 + 1))
    a3 = np.reshape(outs[0], (1, 416 // 8, 416 // 8, 3, 5 + 1))
    boxes, scores, classes = _yolo_out([a1, a2, a3], image_shape, 416)
    #boxes, box_confidence, box_class_probs = _process_feats(pred, anchors, masks, 416)
    print(boxes)
    print(len(boxes))
    print(scores)
    print(classes)

    return img