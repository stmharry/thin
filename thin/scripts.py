import functools
import numpy as np
import tensorflow as tf


def _with(**kwargs):
    def _with_decorator(f):
        for (key, value) in kwargs.items():
            values = getattr(f, key, [])
            values.insert(0, value)
            setattr(f, key, values)

        return f
    return _with_decorator


def with_input(name, dtype=None, default=None):
    return _with(
        input_names=name,
        input_types=dtype,
        input_defaults=default)


def with_output(name, dtype=None, shape=None):
    return _with(
        output_names=name,
        output_types=dtype,
        output_shapes=shape)


def _as_func(as_=None):
    def _as_func_decorator(f, include_inputs=False, *args, **kwargs):
        f_partial = functools.partial(f, *args, **kwargs)

        input_names = getattr(f, 'input_names', [])
        input_types = getattr(f, 'input_types', [])
        input_defaults = getattr(f, 'input_defaults', [])

        output_names = getattr(f, 'output_names', [])
        output_types = getattr(f, 'output_types', [])
        output_shapes = getattr(f, 'output_shapes', [])

        def _get_output_list(inputs):
            input_list = []
            for (name, dtype, default) in zip(input_names, input_types, input_defaults):
                if name in inputs:
                    input_ = inputs[name]
                elif as_ in ['np_fn', 'np_gen']:
                    input_ = np.asarray(default)
                elif as_ in ['tf', 'tf_map']:
                    input_ = tf.constant(default, dtype=dtype)

                input_list.append(input_)

            if as_ in ['np_fn', 'np_gen']:
                output_list = f_partial(*input_list)
            elif as_ == 'tf':
                output_list = tf.numpy_function(f_partial, input_list, output_types)
            elif as_ == 'tf_map':
                output_list = tf.map_fn(
                    lambda _input_list:
                        tf.numpy_function(f_partial, _input_list, output_types),
                    input_list,
                    fn_output_signature=output_types)

            return output_list

        def _set_outputs(inputs, output_list):
            if include_inputs:
                outputs = inputs
            else:
                outputs = {}

            for (output, name, shape) in zip(output_list, output_names, output_shapes):
                if as_ in ['np_fn', 'np_gen']:
                    outputs[name] = output
                elif as_ == 'tf':
                    outputs[name] = tf.ensure_shape(output, shape)
                elif as_ == 'tf_map':
                    outputs[name] = tf.ensure_shape(output, (None,) + tuple(shape))

            return outputs

        # writing separate functions so tf can parse
        if as_ == 'np_gen':
            def _f(inputs={}):
                output_list = _get_output_list(inputs)
                for _output_list in output_list:
                    yield _set_outputs(inputs, _output_list)
        else:
            def _f(inputs={}):
                output_list = _get_output_list(inputs)
                return _set_outputs(inputs, output_list)

        if not include_inputs:
            _f.output_shapes = dict(zip(output_names, output_shapes))
            _f.output_types = dict(zip(output_names, output_types))

        return _f
    return _as_func_decorator


as_numpy_func = _as_func(as_='np_fn')
as_numpy_gen = _as_func(as_='np_gen')
as_tensorflow_func = _as_func(as_='tf')
as_tensorflow_map_func = _as_func(as_='tf_map')
