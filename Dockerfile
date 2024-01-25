FROM intelpython/intelpython3_full:2020.1

RUN pip install pooch pillow toml pywavelets torch jupyter
# numpy scipy tqdm matplotlib

COPY . imagelab

RUN pip install ./imagelab

ENTRYPOINT ["ipython"]