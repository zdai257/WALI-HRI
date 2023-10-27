#!/bin/sh
mkdir -p data

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1sxh_G570LDugvOscMBNTwAHXy8araRMW' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1sxh_G570LDugvOscMBNTwAHXy8araRMW" -O data/sound_recording.wav && rm -rf /tmp/cookies.txt
