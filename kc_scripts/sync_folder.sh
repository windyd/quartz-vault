src="/Users/kevin/Documents/obsidianVault/My Life/"

# the parent folder of this shell script
proj_folder=$(dirname $(dirname $(realpath $0)))
dst="${proj_folder}/content/My Life/"
echo "Syncing $src to $dst"

rsync -av --delete "$src" "$dst"
