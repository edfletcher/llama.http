#!/bin/bash

REPO_PATH=$1
TARGET_PATH=$2
HF_USER_NAME=$3
MODEL_REPO=$4
MODEL_FILE_NAME=$5
DISPLAY_NAME=$6
DESCRIPTION=$7

if [ -z ${MODEL_REPO} ]; then
	echo -n "${HF_USER_NAME} repository name> "
	read MODEL_REPO
fi

if [ -z ${DISPLAY_NAME} ]; then
	echo -n "Display name> "
	read DISPLAY_NAME
fi

if [ -z ${DESCRIPTION} ]; then
	echo -n "Description (optional)> "
	read DESCRIPTION
fi

if [ -z ${MODEL_FILE_NAME} ]; then
	echo -n "Model file name> "
	read MODEL_FILE_NAME
fi

URL_POST="${HF_USER_NAME}/${MODEL_REPO}"
MODEL_REPO_URL="https://huggingface.co/${URL_POST}"
echo "Cloning $MODEL_REPO_URL into $REPO_PATH..."
pushd $REPO_PATH
GIT_LFS_SKIP_SMUDGE=1 git clone $MODEL_REPO_URL

echo "Pulling ${MODEL_FILE_NAME}..."
pushd $MODEL_REPO
git lfs pull --include $MODEL_FILE_NAME
popd; popd;

REPO_LINK_PATH=${REPO_PATH/$TARGET_PATH/}
echo "Linking $REPO_LINK_PATH/$MODEL_REPO/$MODEL_FILE_NAME to $TARGET_PATH/$MODEL_FILE_NAME"
ln -s $REPO_LINK_PATH/$MODEL_REPO/$MODEL_FILE_NAME $TARGET_PATH/$MODEL_FILE_NAME

echo "{\"displayName\":\"${DISPLAY_NAME}\",\"sourceURL\":\"${MODEL_REPO_URL}\",\"description\":\"${DESCRIPTION}\"}" | jq . > $TARGET_PATH/$MODEL_FILE_NAME.json
