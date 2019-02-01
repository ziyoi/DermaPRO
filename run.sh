#!/bin/bash

#  -p 6006:6006
docker run --rm -it --privileged \
  -v "$PWD":/tensorflow/work \
  -v "$HOME/data/molenet":/tensorflow/data \
  -v /dev/bus/usb:/dev/bus/usb \
  detect-tf
