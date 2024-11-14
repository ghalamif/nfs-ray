import os
import time
import ray
import dask.dataframe as dd
from pyspark.sql import SparkSession
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import psutil
import platform
from threading import Thread

# Path to Stanford Dogs Dataset images
imagenet_path = "/srv/nfs/kube-ray/benchmark/Images"
output_dir = "/tmp/processed_images"
os.makedirs(output_dir, exist_ok=True)

# Recursive function to get all image paths in subdirectories
def get_all_image_paths(root_dir):
    image_paths = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(root, file))
    return image_paths

# List of all image file paths
all_image_paths = get_all_image_paths(imagenet_path)

# Verify that image paths were found
print("Number of images found:", len(all_image_paths))  # Should be greater than zero
print("Sample image paths:", all_image_paths[:5])

# Image processing function
def load_and_process_image(file_path):
    img = Image.open(file_path)
    img = img.resize((256, 256)).convert("L")  # Resize and convert to grayscale
    return img

# Resource monitoring function for current process
def track_resource_usage(interval, stop_flag, cpu_list, ram_list):
    process = psutil.Process(os.getpid())
    while not stop_flag["stop"]:
        cpu_percent = process.cpu_percent(interval=None)
        ram_usage = process.memory_info().rss / (1024 ** 3)  # Convert to GB
        cpu_list.append(cpu_percent)
        ram_list.append(ram_usage)
        time.sleep(interval)

# Benchmark function for Ray with resource tracking
def benchmark_ray_data():
    stop_flag = {"stop": False}
    cpu_usage = []
    ram_usage = []

    # Start the resource tracking thread
    monitor_thread = Thread(target=track_resource_usage, args=(0.1, stop_flag, cpu_usage, ram_usage))  # 100 ms interval
    monitor_thread.start()

    ray.init()
    start_time = time.time()

    ds = ray.data.from_items([{"image_path": path} for path in all_image_paths])
    ds = ds.map_batches(lambda df: {"processed_images": [load_and_process_image(file) for file in df["image_path"]]})
    ds.show()  # Trigger computation

    ray.shutdown()
    total_time = time.time() - start_time

    # Stop the resource tracking
    stop_flag["stop"] = True
    monitor_thread.join()

    return total_time, cpu_usage, ram_usage

# Benchmark function for Dask with resource tracking
def benchmark_dask():
    stop_flag = {"stop": False}
    cpu_usage = []
    ram_usage = []

    # Start the resource tracking thread
    monitor_thread = Thread(target=track_resource_usage, args=(0.1, stop_flag, cpu_usage, ram_usage))  # 100 ms interval
    monitor_thread.start()

    start_time = time.time()
    df = dd.from_pandas(pd.DataFrame({"image_path": all_image_paths}), npartitions=4)
    df["processed"] = df["image_path"].apply(load_and_process_image, meta=('str'))
    df.compute()  # Trigger computation
    total_time = time.time() - start_time

    # Stop the resource tracking
    stop_flag["stop"] = True
    monitor_thread.join()

    return total_time, cpu_usage, ram_usage

# Benchmark function for Spark with resource tracking
def benchmark_spark():
    stop_flag = {"stop": False}
    cpu_usage = []
    ram_usage = []

    # Start the resource tracking thread
    monitor_thread = Thread(target=track_resource_usage, args=(0.1, stop_flag, cpu_usage, ram_usage))  # 100 ms interval
    monitor_thread.start()

    spark = (SparkSession.builder
             .appName("ImageNet Benchmarking")
             .config("spark.driver.bindAddress", "127.0.0.1")  # Bind to localhost
             .config("spark.driver.memory", "8g")  # Increase driver memory
             .config("spark.executor.memory", "8g")  # Increase executor memory
             .getOrCreate())
    start_time = time.time()

    df = spark.createDataFrame(pd.DataFrame({"image_path": all_image_paths}))

    # Process and save images without using collect()
    def process_and_save(row):
        img = load_and_process_image(row["image_path"])
        img.save(os.path.join(output_dir, os.path.basename(row['image_path'])))

    df.rdd.foreach(process_and_save)  # Distributed processing without collecting all

    spark.stop()
    total_time = time.time() - start_time

    # Stop the resource tracking
    stop_flag["stop"] = True
    monitor_thread.join()

    return total_time, cpu_usage, ram_usage

# Run benchmarks if images are found
if all_image_paths:
    ray_time, ray_cpu_usage, ray_ram_usage = benchmark_ray_data()
    dask_time, dask_cpu_usage, dask_ram_usage = benchmark_dask()
    spark_time, spark_cpu_usage, spark_ram_usage = benchmark_spark()

    # Processing time comparison plot
    frameworks = ['Ray', 'Dask', 'Spark']
    times = [ray_time, dask_time, spark_time]

    plt.figure(figsize=(15, 15))

    plt.subplot(3, 1, 1)
    plt.bar(frameworks, times, color=['blue', 'orange', 'green'])
    plt.ylabel('Processing Time (seconds)')
    plt.title('Processing Time Comparison')

    # Resource usage over time plot
    time_interval = 0.1  # 100 ms interval

    # Plot CPU usage
    time_points_ray = [i * time_interval for i in range(len(ray_cpu_usage))]
    time_points_dask = [i * time_interval for i in range(len(dask_cpu_usage))]
    time_points_spark = [i * time_interval for i in range(len(spark_cpu_usage))]

    plt.subplot(3, 1, 2)
    plt.plot(time_points_ray, ray_cpu_usage, label='Ray', color='blue')
    plt.plot(time_points_dask, dask_cpu_usage, label='Dask', color='orange')
    plt.plot(time_points_spark, spark_cpu_usage, label='Spark', color='green')
    plt.xlabel('Time (seconds)')
    plt.ylabel('CPU Usage (%)')
    plt.title('CPU Usage Over Time')
    plt.legend()

    # Plot RAM usage
    plt.subplot(3, 1, 3)
    plt.plot(time_points_ray, ray_ram_usage, label='Ray', color='blue')
    plt.plot(time_points_dask, dask_ram_usage, label='Dask', color='orange')
    plt.plot(time_points_spark, spark_ram_usage, label='Spark', color='green')
    plt.xlabel('Time (seconds)')
    plt.ylabel('RAM Usage (GB)')
    plt.title('RAM Usage Over Time')
    plt.legend()

    plt.tight_layout()

    # Save the plot as a PDF
    plt.savefig('benchmark_results.pdf', format='pdf')
    print("Benchmark plot saved as 'benchmark_results.pdf'")

else:
    print("No images found in the specified directory. Please check the path and directory structure.")
