# Checking all nodes to have gpu and cuda available
for node in $(kubectl get nodes -o jsonpath='{.items[*].metadata.name}'); do
    echo "Checking nvidia-smi on $node:"
    kubectl run gpu-check --rm -t -i --restart=Never --image=nvcr.io/nvidia/k8s/cuda-sample:vectoradd-cuda11.7.1-ubuntu20.04 --overrides='{"spec": {"nodeName": "'$node'"}}' -- nvidia-smi
    echo "##########################################################################"
done
