#!/bin/bash

PATH="/usr/lib/jvm/java-8-openjdk-amd64/bin":$PATH
JAVA_HOME="/usr/lib/jvm/java-8-openjdk-amd64"

mvn clean package
