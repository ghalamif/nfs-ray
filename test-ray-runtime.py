import ray

runtime_env = {
    "pip": [
        "torch",
        "torchvision",
        "pandas",
        "scikit-learn",
        "ray[tune]",
        "ray[data]",
        "ray[default]"
    ]
}

ray.init(address="auto", runtime_env=runtime_env)


@ray.remote
def f():

    import ray.tune
    import ray.data
    import ray.train
    print(torch.__version__)
    print(torchvision.__version__)
  

print(ray.get(f.remote()))
