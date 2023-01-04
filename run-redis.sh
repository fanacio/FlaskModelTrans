#!/bin/bash
if [ ! -d redis-stable/src ]; then
    wget http://download.redis.io/redis-stable.tar.gz --no-check-certificate && tar xvzf redis-stable.tar.gz && rm redis-stable.tar.gz
fi
cd redis-stable
make
src/redis-server
