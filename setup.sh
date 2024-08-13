#!/bin/bash

namespace="kube-ray"
service_account="kuberay-service-account"
ray_cluster_value_path="kuube-ray/ray-cluster-values.yaml"

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
    helm install kuberay kuberay/kuberay-operator --namespace $namespace    
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
        helm install ray-cluster kuberay/ray-cluster -f $ray_cluster_value_path --namespace $namespace

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

################################################################################
# Call the function to install the NVIDIA device plugin for k8s
################################################################################
install_nvidia_device_plugin

################################################################################
# Call the function to create the ray operator in the specified namespace
################################################################################
create_kube_ray_operator

################################################################################
# Call the function to create the ray cluster in the specified namespace
################################################################################
create_ray_cluster
