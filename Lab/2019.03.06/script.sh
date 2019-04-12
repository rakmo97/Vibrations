#!/bin/sh

source /opt/asn/etc/asn-bash-profiles-special/modules.sh

singularity exec shub://nuitrcs/fenics python3 tensile_time.py
