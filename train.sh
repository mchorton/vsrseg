#!/bin/sh
python vsrseg/main.py train -e 6 -l 0.00001 --cuda 0 --save_per=2
