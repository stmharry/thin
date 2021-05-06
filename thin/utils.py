import collections
import functools
import gin
import numpy as np
import tensorflow as tf


def on_cpu(f):
    @functools.wraps(f)
    def _f(*args, **kwargs):
        with tf.device('CPU'):
            return f(*args, **kwargs)
    return _f


def logged_tensor(tensor, name=None):
    name = name or tensor.name

    print_op = tf.print(f'{name}:', tensor)
    with tf.control_dependencies([print_op]):
        return tf.identity(tensor)


def maybe_decode(s):
    if isinstance(s, bytes):
        s = s.decode()
    return s


def ndarray_feature(x):
    x = np.asarray(x).flatten()

    if x.dtype.type == np.str_:
        x = x.astype('S')
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=x))
    elif x.dtype in [np.float32, np.float64]:
        return tf.train.Feature(float_list=tf.train.FloatList(value=x))
    elif x.dtype in [np.int32, np.int64]:
        return tf.train.Feature(int64_list=tf.train.Int64List(value=x))
    else:
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[x.tobytes()]))


def ndarray_from_feature(x, dtype, shape=None):
    x = tf.io.decode_raw(x, dtype)
    if shape is not None:
        x = tf.reshape(x, shape)
    return x


def namedtuple(typename, field_defaults):
    cls = collections.namedtuple(typename, field_defaults.keys())
    cls.__new__.__defaults__ = tuple(field_defaults.values())
    return cls


def gin_query_parameter(key, default=None):
    try:
        return gin.query_parameter(key)
    except ValueError:
        return default
