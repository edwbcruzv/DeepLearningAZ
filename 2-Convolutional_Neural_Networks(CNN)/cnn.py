#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 20:20:35 2024

@author: cruz
"""

# =============================================================================
# Constuccion del modelo de CNN
# =============================================================================
# Importacion de librerias
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Inicializando la CNN
classifier = Sequential()

# =============================================================================
# PASO 1: Convolucion
# =============================================================================
classifier.add(Convolution2D(filters=32,
                             kernel_size=(3,3),
                             input_shape= (64,64,3),
                             activation="relu") )
# =============================================================================
# PASO 2: Max Pooling
# =============================================================================
classifier.add(MaxPooling2D(pool_size=(2,2)))
# =============================================================================
# PASO 3: Flattening
# =============================================================================
classifier.add(Flatten())
# =============================================================================
# PASO 4: Full Connection
# =============================================================================
classifier.add(Dense(units=128, activation="relu"))
classifier.add(Dense(units=1, activation="sigmoid"))
# =============================================================================
# Compilando la red neuronal de convolucion
# =============================================================================
classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])