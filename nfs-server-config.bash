#!/bin/bash

# Define the NFS directory
NFS_DIR="/srv/nfs/kube-ray"

# Step 1: Set Directory Permissions to 777
echo "Setting permissions on $NFS_DIR to 777..."
sudo chmod -R 777 $NFS_DIR

# Step 2: Check and Remove Duplicate Entries in /etc/exports
echo "Checking for duplicate entries in /etc/exports..."
EXPORTS_LINE="$NFS_DIR *(rw,sync,no_subtree_check,all_squash,anonuid=1000,anongid=1000)"

# Remove all entries matching the directory, regardless of options
sudo sed -i "\|$NFS_DIR|d" /etc/exports

# Add the correct export line
echo "$EXPORTS_LINE" | sudo tee -a /etc/exports
echo "Export line added to /etc/exports."

# Step 3: Reapply Export Configuration and Restart NFS Service
echo "Reapplying export configuration..."
sudo exportfs -ra

echo "Restarting NFS server..."
sudo systemctl restart nfs-server

# Step 4: Check the status of the NFS server
if sudo systemctl status nfs-server | grep -q "active (running)"; then
    echo "NFS server is running successfully."
else
    echo "NFS server is not running. Checking for issues..."

    # Check NFS service status
    sudo systemctl status nfs-server

    # Check for NFS-related logs
    echo "Checking NFS logs..."
    sudo journalctl -u nfs-server | tail -n 20

    echo "Checking NFS exports..."
    sudo exportfs -v

    echo "Please manually inspect the above output for errors."
    exit 1
fi

# Step 5: Ensure RPC and related services are running
echo "Ensuring related services are running..."
sudo systemctl restart rpcbind
sudo systemctl restart nfs-mountd
sudo systemctl restart nfs-server

# Check that all necessary services are active
echo "Verifying NFS-related services status..."
sudo systemctl status rpcbind
sudo systemctl status nfs-mountd
sudo systemctl status nfs-server

# Final check on the NFS server status
if sudo systemctl status nfs-server | grep -q "active (running)"; then
    echo "NFS server and related services are running successfully."
else
    echo "Failed to start NFS server or related services. Please check the logs for more details."
    exit 1
fi

echo "NFS server configuration complete."
