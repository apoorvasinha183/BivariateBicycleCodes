#!/bin/bash
>debug.txt
for x in {0..143}
do
    python defect_based_decoder_setup.py -err 0.0037 -repair asym -arg $x >> debug.txt
done
