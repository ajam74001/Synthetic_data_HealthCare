import pandas as pd
import numpy as np
import torch

from gretel_synthetics.timeseries_dgan.dgan import DGAN
from gretel_synthetics.timeseries_dgan.config import DGANConfig, OutputType

#TODO: add maybe the attributes as well (demographics) - 
# %%capture
data = pd.read_csv("/userfiles/ajamshidi18/PCG_timeseris_normal.csv")
print(data.isna().sum().sum())

features = np.array(data).reshape(data.shape[0], data.shape[1], 1)

# Train DGAN model
# sample_len :time series steps to generate from each LSTM cell in DGAN
# TODO: loging of the performance 
model = DGAN(DGANConfig(
    max_sequence_len=features.shape[1],
    sample_len=1489,
    batch_size=min(1000, features.shape[0]),
    apply_feature_scaling=True,
    apply_example_scaling=False,
    use_attribute_discriminator=False,
   
    generator_learning_rate=1e-4,
    discriminator_learning_rate=1e-4,
    epochs=10000,
))
# feature_types=[OutputType.CONTINUOUS] * features.shape[2],
model.train_numpy(
    features,
)

# Generate synthetic data
_, synthetic_features = model.generate_numpy(1000)

