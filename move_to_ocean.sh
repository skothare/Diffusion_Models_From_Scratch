#!/bin/bash
# Run this from /jet/home/skothare/F25-Deep-Learning-Project

cd /jet/home/skothare/F25-Deep-Learning-Project

# 1) Create directories on the project (/ocean) space
mkdir -p /ocean/projects/cis250280p/skothare/{data,experiments,pretrained}

# 2) Move and symlink the large directories
for dir in data experiments pretrained; do
    if [ -d "$dir" ]; then
        echo "Moving $dir to /ocean/projects/cis250280p/skothare/$dir ..."
        mv "$dir" /ocean/projects/cis250280p/skothare/
        ln -s /ocean/projects/cis250280p/skothare/"$dir" "$dir"
        echo "✓ $dir moved and symlinked"
    else
        echo "Skipping $dir (directory not found)"
    fi
done

# 3) Show how much home space you're using now
echo ""
echo "Home directory usage:"
df -h /jet/home/skothare
