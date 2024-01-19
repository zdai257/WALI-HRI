#!/bin/sh

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1lxS4fjqs31SIq5al9KxFSRbs0Vqq8ti4' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1lxS4fjqs31SIq5al9KxFSRbs0Vqq8ti4" -O WALI-HRI_sample.zip && rm -rf /tmp/cookies.txt
