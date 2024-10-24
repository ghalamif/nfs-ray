#!/bin/bash

namespace="kube-ray"
service_account="kuberay-service-account"
ray_cluster_value_path="kube-ray/ray-cluster-values.yaml"
manual=0
manual_cluster_path="k8s/kube-ray/ray-cluster.yaml"

install_nvidia_device_plugin() {

    # Check if the 'gpu-operator' namespace exists
    if ! kubectl get ns gpu-operator >/dev/null 2>&1; then
        echo "Creating 'gpu-operator' namespace."
        kubectl create ns gpu-operator
    else
        echo "'gpu-operator' namespace already exists."
    fi

    # Label the 'gpu-operator' namespace with the necessary security context
    echo "Labeling 'gpu-operator' namespace for privileged pod security."
    kubectl label --overwrite ns gpu-operator pod-security.kubernetes.io/enforce=privileged

    # Add NVIDIA Helm repository if it doesn't exist
    if ! helm repo list | grep -q "nvidia"; then
        echo "Adding NVIDIA Helm repository."
        helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
    else
        echo "NVIDIA Helm repository already added."
    fi

    # Update the Helm repository to ensure we have the latest charts
    echo "Updating Helm repository."
    helm repo update

    # Install the NVIDIA GPU Operator if it's not already installed
    if ! helm ls -n gpu-operator | grep -q "gpu-operator"; then
        echo "Installing NVIDIA GPU Operator."
        helm install --wait --generate-name nvidia/gpu-operator --set operator.defaultRuntime="containerd" -n gpu-operator
    else
        echo "NVIDIA GPU Operator already installed."
    fi

    echo "NVIDIA device plugin installation complete."
}

create_namespace() {
    # Check if the namespace already exists and create it if it doesn't
    if kubectl get namespace $namespace &> /dev/null; then
        echo "Namespace $namespace already exists. Skipping."
    else
        echo "Creating namespace $namespace."
        kubectl create namespace $namespace
    fi
}

create_kube_ray_operator() {
    # Check if the namespace already exists and create it if it doesn't
    create_namespace

    # Check if the service account already exists and create it if it doesn't
    if kubectl get serviceaccount $service_account -n $namespace &> /dev/null; then
        echo "Service account $service_account already exists. Skipping."
    else
        echo "Creating service account $service_account."
        kubectl apply -f service-account-for-ray.yaml
    fi

    # Create the ray-operator
    helm repo add kuberay https://ray-project.github.io/kuberay-helm/
    helm repo update
    sleep 3
    helm install kuberay-operator kuberay/kuberay-operator --version 1.1.1 --namespace $namespace 
    sleep 3
}

create_ray_cluster() {
    # Check if the namespace already exists and create it if it doesn't
    create_namespace

    if helm list -n $namespace | grep -q ray-cluster; then
        echo "Ray cluster already exists. Upgrading."
        upgrade_ray_cluster
        return
    else
        echo "Creating Ray cluster."
        if [ manual==1 ]; then
            kubectl apply -f $manual_cluster_path
        else
            helm install raycluster kuberay/ray-cluster -f $ray_cluster_value_path --namespace $namespace
        fi
        helm install raycluster kuberay/ray-cluster -f $ray_cluster_value_path --namespace $namespace
        sleep 3
        # Get the ray cluster status
        kubectl get raycluster -n $namespace
    fi
}

upgrade_ray_cluster() {
    helm upgrade ray-cluster kuberay/ray-cluster -f $ray_cluster_value_path --namespace $namespace
}

delete_kube_ray_operator_and_cluster() {
    # Delete the Ray cluster if it exists
    if helm list -n $namespace | grep -q ray-cluster; then
        echo "Deleting Ray cluster."
        helm uninstall ray-cluster --namespace $namespace
    else
        echo "Ray cluster not found. Skipping."
    fi

    # Delete the Ray operator if it exists
    if helm list -n $namespace | grep -q kuberay; then
        echo "Deleting KubeRay operator."
        helm uninstall kuberay --namespace $namespace
    else
        echo "KubeRay operator not found. Skipping."
    fi

    # Delete the service account if it exists
    if kubectl get serviceaccount $service_account -n $namespace &> /dev/null; then
        echo "Deleting service account $service_account."
        kubectl delete serviceaccount $service_account -n $namespace
    else
        echo "Service account $service_account not found. Skipping."
    fi

    # Delete the namespace if it exists
    if kubectl get namespace $namespace &> /dev/null; then
        echo "Deleting namespace $namespace."
        kubectl delete namespace $namespace
    else
        echo "Namespace $namespace not found. Skipping."
    fi
}




if [ "$1" == "nvidia" ]; then

    install_nvidia_device_plugin

elif [ "$1" == "create" ]; then
    if [ "$2" == "man" ]; then
        manual=1
    fi
    manual=0
    create_kube_ray_operator
    create_ray_cluster
    

elif [ "$1" == "cluster-update" ]; then

    upgrade_ray_cluster

elif [ "$1" == "destroy" ]; then

    delete_kube_ray_operator_and_cluster

else
    echo "Usage: $0 {create [--static]|destroy|list|list-ips}"
fi