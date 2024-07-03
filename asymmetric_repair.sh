#!/bin/bash
> ckt_asym.txt
python defect_based_decoder_setup.py -err 0.001 -repair asym >> ckt_asym.txt
python defect_based_decoder_setup.py -err 0.002 -repair asym >> ckt_asym.txt
python defect_based_decoder_setup.py -err 0.003 -repair asym >> ckt_asym.txt
python defect_based_decoder_setup.py -err 0.004 -repair asym >> ckt_asym.txt
python defect_based_decoder_setup.py -err 0.005 -repair asym >> ckt_asym.txt
python defect_based_decoder_setup.py -err 0.006 -repair asym >> ckt_asym.txt
python defect_based_decoder_setup.py -err 0.007 -repair asym >> ckt_asym.txt