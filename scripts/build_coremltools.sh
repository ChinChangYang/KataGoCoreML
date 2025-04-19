if [ -d "coremltools" ]; then
    echo "[INFO] coremltools already exists. Reusing local directory."
else
    git clone https://github.com/apple/coremltools.git
fi

cd coremltools
git checkout 8.2
zsh -i scripts/env_create.sh --python=3.11 --exclude-test-deps
zsh -i scripts/build.sh --python=3.11 --no-check-env
