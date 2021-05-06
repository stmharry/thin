import gin
import imgaug.augmenters as iaa

from tensorflow.compat.v1.keras.layers import BatchNormalization
from tensorflow_addons.layers import GroupNormalization
from tensorflow_addons.layers import InstanceNormalization

gin.config.external_configurable(BatchNormalization, module='tf.keras.layers')
gin.config.external_configurable(GroupNormalization, module='tf.keras.layers')
gin.config.external_configurable(InstanceNormalization, module='tf.keras.layers')

for (iaa_object_name, iaa_object) in iaa.__dict__.items():
    if isinstance(iaa_object, type) and issubclass(iaa_object, iaa.Augmenter):
        gin.external_configurable(iaa_object, module='iaa')
