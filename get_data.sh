#! /bin/bash
mkdir -p /opt/data
cd /opt/data
#wget https://doc-0s-8c-docs.googleusercontent.com/docs/securesc/ha0ro937gcuc7l7deffksulhg5h7mbp1/0foqmn7ffvrc0s77450bc8u7aomftvoa/1552132800000/11393535061093459801/*/1OEJSzj_TBCgBpDgh-BL0dQgbLgU7cOFi?e=download
#mv 1OEJSzj_TBCgBpDgh-BL0dQgbLgU7cOFi?e=download own_data.zip
unzip -n /home/workspace/own_data.zip -d /opt/data/own_data/
wget https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip
unzip -n /home/workspace/own_data.zip -d /opt/data/udacity_data/
unzip -n /home/workspace/more_data.zip -d /opt/data/more_data/
mkdir /opt/data/IMG
mv /opt/data/own_data/data/IMG/*  /opt/data/IMG/
mv /opt/data/udacity_data/data/IMG/*  /opt/data/IMG/
mv /opt/data/more_data/data/IMG/*  /opt/data/IMG/
ls -ltrah

