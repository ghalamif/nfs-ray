#!/bin/bash

# Define the NFS server and directory
NFS_SERVER="192.168.209.37"
NFS_DIR="/srv/nfs/kube-ray"

# Step 1: Ensure nfs-common is installed
echo "Checking if nfs-common is installed..."
if ! dpkg -l | grep -qw nfs-common; then
    echo "nfs-common is not installed. Installing it now..."
    sudo apt update
    sudo apt install nfs-common -y
else
    echo "nfs-common is already installed."
fi

# Step 2: Check if the NFS server is reachable
echo "Checking if NFS server $NFS_SERVER is reachable..."
if showmount -e $NFS_SERVER; then
    echo "NFS server is reachable and exports are listed above."
else
    echo "Failed to reach NFS server $NFS_SERVER. Please check the network connection and NFS server status."
    exit 1
fi

# Step 3: Check if the NFS directory exists, otherwise create it
if [ ! -d "$NFS_DIR" ]; then
    echo "NFS directory $NFS_DIR does not exist. Creating it..."
    sudo mkdir -p $NFS_DIR
    echo "Directory $NFS_DIR created."
else
    echo "NFS directory $NFS_DIR already exists."
fi

# Step 4: Check if the NFS directory is already mounted
if mountpoint -q $NFS_DIR; then
    echo "NFS directory $NFS_DIR is already mounted."
else
    # Mount the NFS directory
    echo "Mounting the NFS directory from $NFS_SERVER..."
    sudo mount -t nfs $NFS_SERVER:$NFS_DIR $NFS_DIR -o rw

    # Add the mount to /etc/fstab for automatic mounting on boot
    FSTAB_LINE="$NFS_SERVER:$NFS_DIR $NFS_DIR nfs defaults,rw 0 0"
    if grep -Fxq "$FSTAB_LINE" /etc/fstab; then
        echo "NFS mount already configured in /etc/fstab."
    else
        echo "$FSTAB_LINE" | sudo tee -a /etc/fstab
        echo "NFS mount added to /etc/fstab."
    fi
fi

# Step 5: Set umask for all users to 0000
echo "Setting umask for all users to 0000..."
UMASK_LINE="umask 0000"
BASHRC_PATH="/etc/bash.bashrc"  # Global bashrc file for all users on most systems

if grep -Fxq "$UMASK_LINE" $BASHRC_PATH; then
    echo "umask 0000 already set in $BASHRC_PATH."
else
    echo "$UMASK_LINE" | sudo tee -a $BASHRC_PATH
    echo "umask 0000 added to $BASHRC_PATH."
fi

echo "NFS client configuration complete."
